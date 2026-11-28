#line 1 "Source/RGBMoCapMetaHuman/Private/RGBMoCapFace.cpp"
#include "RGBMoCapConnectComponent.h"
#include "RGBMoCapFaceAnimInstance.h"
#include "Animation/PoseAsset.h"
#include "GameFramework/Actor.h"
#include "Features/IModularFeatures.h"
#include "ILiveLinkClient.h"
#include "Roles/LiveLinkAnimationRole.h"
#include "Roles/LiveLinkAnimationTypes.h"

namespace {
const TCHAR* FaceChannels[52]={TEXT("BrowDownLeft"),TEXT("BrowDownRight"),TEXT("BrowInnerUp"),TEXT("BrowOuterUpLeft"),TEXT("BrowOuterUpRight"),TEXT("CheekPuff"),TEXT("CheekSquintLeft"),TEXT("CheekSquintRight"),TEXT("EyeBlinkLeft"),TEXT("EyeBlinkRight"),TEXT("EyeLookDownLeft"),TEXT("EyeLookDownRight"),TEXT("EyeLookInLeft"),TEXT("EyeLookInRight"),TEXT("EyeLookOutLeft"),TEXT("EyeLookOutRight"),TEXT("EyeLookUpLeft"),TEXT("EyeLookUpRight"),TEXT("EyeSquintLeft"),TEXT("EyeSquintRight"),TEXT("EyeWideLeft"),TEXT("EyeWideRight"),TEXT("JawForward"),TEXT("JawLeft"),TEXT("JawOpen"),TEXT("JawRight"),TEXT("MouthClose"),TEXT("MouthDimpleLeft"),TEXT("MouthDimpleRight"),TEXT("MouthFrownLeft"),TEXT("MouthFrownRight"),TEXT("MouthFunnel"),TEXT("MouthLeft"),TEXT("MouthLowerDownLeft"),TEXT("MouthLowerDownRight"),TEXT("MouthPressLeft"),TEXT("MouthPressRight"),TEXT("MouthPucker"),TEXT("MouthRight"),TEXT("MouthRollLower"),TEXT("MouthRollUpper"),TEXT("MouthShrugLower"),TEXT("MouthShrugUpper"),TEXT("MouthSmileLeft"),TEXT("MouthSmileRight"),TEXT("MouthStretchLeft"),TEXT("MouthStretchRight"),TEXT("MouthUpperUpLeft"),TEXT("MouthUpperUpRight"),TEXT("NoseSneerLeft"),TEXT("NoseSneerRight"),TEXT("TongueOut")};
USkeletalMeshComponent* FaceMesh(AActor* Owner) {
    if (!Owner) return nullptr;
    TArray<USkeletalMeshComponent*> Meshes;Owner->GetComponents(Meshes);
    for (auto* M:Meshes) if (M->GetFName()==TEXT("Face")) return M;
    return nullptr;
}
ILiveLinkClient* FaceClient() {
    auto& F=IModularFeatures::Get();return F.IsModularFeatureAvailable(ILiveLinkClient::ModularFeatureName)?&F.GetModularFeature<ILiveLinkClient>(ILiveLinkClient::ModularFeatureName):nullptr;
}
bool FreshFace(const FLiveLinkSubjectFrameData& Frame) {
    auto* D=Frame.FrameData.Cast<FLiveLinkAnimationFrameData>();if (!D) return false;
    double Age=FPlatformTime::Seconds()-(D->ArrivalTime.WorldTime>0.?D->ArrivalTime.WorldTime:D->WorldTime.GetOffsettedTime());
    return FMath::IsFinite(Age) && Age>=-1. && Age<=2.;
}
}
void URGBMoCapConnectComponent::SetFaceEnabled(bool Value) {
    const bool Enable=SupportsFace() && Value;
    if (FaceEnabled!=Enable) {Modify();FaceEnabled=Enable;}
    if (!Enable && (SupportsFace() || FaceHasBackup)) RestoreFace();
    if (!Enable) FaceStatus=SupportsFace()?TEXT("Disabled"):TEXT("Not available for SK Mannequin");
}
void URGBMoCapConnectComponent::RestoreFace() {
    auto* Face=FaceMesh(GetOwner());
    // Validate captured component identity; reconstruction may leave TRASH
    // objects alive while copying the native override onto the replacement.
    if (FaceHasBackup && Face) {
        Face->Modify(false);
        if (Face->GetAnimClass()==URGBMoCapFaceAnimInstance::StaticClass()) {Face->SetAnimInstanceClass(SavedFaceAnimClass);Face->SetAnimationMode(SavedFaceMode);}
        if (Face->VisibilityBasedAnimTickOption==EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones) Face->VisibilityBasedAnimTickOption=SavedFaceVisibilityTick;
#if WITH_EDITOR
        Face->SetUpdateAnimationInEditor(SavedFaceEditorTick);
#endif
    }
    FaceHasBackup=false;FaceCurves.Reset();FaceCurveNames.Reset();FaceMatrix.Reset();MappingPose=nullptr;
    FaceSubject=NAME_None;FaceProvider.Invalidate();
    MaintainPreviewTicks();
}
bool URGBMoCapConnectComponent::PrepareFaceMapping() {
    if (MappingPose && !FaceMatrix.IsEmpty()) return true;
    MappingPose=FaceMappingPose.LoadSynchronous();
    if (!MappingPose) for (const TCHAR* Path:{
        TEXT("/Game/MetaHumans/Common/Face/ARKit/PA_MetaHuman_ARKit_Mapping.PA_MetaHuman_ARKit_Mapping"),
        TEXT("/Game/MetaHumans/Common/Common/Mocap/mh_arkit_mapping_pose.mh_arkit_mapping_pose")}) {
        MappingPose=LoadObject<UPoseAsset>(nullptr,Path);if (MappingPose) break;
    }
    if (!MappingPose || !MappingPose->IsValidAdditive()) {FaceStatus=TEXT("MetaHuman additive ARKit mapping pose asset is missing. Set Face Mapping Pose on this component.");MappingPose=nullptr;return false;}
    FaceCurveNames=MappingPose->GetCurveFNames();const auto& Poses=MappingPose->GetPoseFNames();
    if (FaceCurveNames.IsEmpty()) {FaceStatus=TEXT("Face mapping asset has no expression curves.");return false;}
    FaceMatrix.Init(0.f,52*FaceCurveNames.Num());int32 Found=0;
    for (int32 Channel=0;Channel<52;++Channel) {
        int32 Pose=Poses.IndexOfByKey(FName(FaceChannels[Channel]));if (Pose<0) continue;
        auto Values=MappingPose->GetCurveValues(Pose);
        if (Values.Num()!=FaceCurveNames.Num()) {FaceStatus=TEXT("Invalid MetaHuman mapping curve data.");FaceMatrix.Reset();return false;}
        for (int32 I=0;I<Values.Num();++I) FaceMatrix[Channel*Values.Num()+I]=Values[I];++Found;
    }
    if (Found<48) {FaceStatus=TEXT("Face mapping does not contain the expected ARKit poses.");FaceMatrix.Reset();return false;}
    return true;
}
bool URGBMoCapConnectComponent::ReadFace() {
    FName Name(*(Subject.ToString()+TEXT("_Face")));
    if (FaceSubject!=Name) {FaceSubject=Name;FaceCurves.Reset();FaceProvider.Invalidate();}
    auto* Client=FaceClient();
    if (!Client) {FaceStatus=TEXT("Waiting for tracked face data");return false;}
    TArray<FGuid> Bodies,Faces;
    for (const auto& K:Client->GetSubjectsSupportingRole(ULiveLinkAnimationRole::StaticClass(),false,false)) {
        if (!Client->IsSubjectEnabled(K,true)) continue;
        if (K.SubjectName==Subject) Bodies.Add(K.Source);
        if (K.SubjectName==Name) Faces.Add(K.Source);
    }
    FLiveLinkSubjectFrameData Body;
    if (Bodies.Num()==1 && Client->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(Bodies[0],Subject),ULiveLinkAnimationRole::StaticClass(),Body) && FreshFace(Body)) {
        auto* D=Body.FrameData.Cast<FLiveLinkAnimationFrameData>();
        const auto* Available=D->MetaData.StringMetaData.Find(TEXT("FaceAvailable"));
        if (Available && *Available==TEXT("0")) {FaceCurves.Reset();FaceStatus=TEXT("No tracked face at this time");return false;}
    }
    if (Bodies.Num()!=1 || Faces.Num()!=1 || Bodies[0]!=Faces[0]) {FaceStatus=TEXT("Waiting for matching tracked face data");return false;}
    if (FaceProvider.IsValid() && FaceProvider!=Faces[0]) FaceCurves.Reset();
    FLiveLinkSubjectFrameData Frame;
    if (!Client->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(Faces[0],Name),ULiveLinkAnimationRole::StaticClass(),Frame) || !FreshFace(Frame)) {FaceStatus=TEXT("Face stream stale - holding last expression");return false;}
    auto* S=Frame.StaticData.Cast<FLiveLinkSkeletonStaticData>();auto* D=Frame.FrameData.Cast<FLiveLinkAnimationFrameData>();
    const auto* OwnerSubject=D?D->MetaData.StringMetaData.Find(TEXT("BodySubject")):nullptr;
    if (!S || !D || !OwnerSubject || *OwnerSubject!=Subject.ToString() || S->PropertyNames.Num()!=52 || D->PropertyValues.Num()!=52) {FaceStatus=TEXT("Invalid face stream schema");return false;}
    const auto* BD=Body.FrameData.Cast<FLiveLinkAnimationFrameData>();
    const FString* BodyRig=BD?BD->MetaData.StringMetaData.Find(TEXT("RigId")):nullptr;
    const FString* FaceRig=D->MetaData.StringMetaData.Find(TEXT("RigId"));
    const FString* BodyFrame=BD?BD->MetaData.StringMetaData.Find(TEXT("SourceFrame")):nullptr;
    const FString* FaceFrame=D->MetaData.StringMetaData.Find(TEXT("MotionFrame"));
    if (!BD || !FreshFace(Body) || !BodyRig || !FaceRig || *BodyRig!=*FaceRig || !BodyFrame || !FaceFrame || *BodyFrame!=*FaceFrame) {
        FaceStatus=TEXT("Waiting for matching body/face frame revisions");return false;
    }
    TArray<float> Curves;Curves.Init(0.f,52);
    for (int32 Channel=0;Channel<52;++Channel) {
        int32 I=S->PropertyNames.IndexOfByKey(FName(FaceChannels[Channel]));
        if (I<0 || !FMath::IsFinite(D->PropertyValues[I])) {FaceStatus=TEXT("Invalid ARKit face coefficients");return false;}
        Curves[Channel]=FMath::Clamp(D->PropertyValues[I],0.f,1.f);
    }
    FaceCurves=MoveTemp(Curves);FaceProvider=Faces[0];
    const auto* Tracked=D->MetaData.StringMetaData.Find(TEXT("FaceTracked"));
    FaceStatus=Tracked && *Tracked==TEXT("0")?TEXT("Streaming - holding last tracked expression"):TEXT("Streaming");return true;
}
void URGBMoCapConnectComponent::UpdateFace() {
    if (!SupportsFace()) {SetFaceEnabled(false);return;}
    auto* Face=FaceMesh(GetOwner());
    // Undo may restore the native preview class after RestoreFace released it.
    // Retain the original baseline rather than capturing that preview as original.
    if (Face && !BackedUpFaceName.IsNone()) {
        TArray<USkeletalMeshComponent*> Current;GetOwner()->GetComponents(Current);
        if (!IsValid(BackedUpFace) || !Current.Contains(BackedUpFace.Get()) || BackedUpFace->GetFName()!=BackedUpFaceName) BackedUpFace=Face;
        if ((Reconcile || FaceHasBackup) && BackedUpFace==Face && Face->GetAnimClass()==URGBMoCapFaceAnimInstance::StaticClass()) FaceHasBackup=true;
    }
    if (!Enabled || !FaceEnabled || Saving) {
        if (FaceHasBackup) RestoreFace();
        FaceStatus=FaceEnabled?TEXT("Waiting for body connection"):TEXT("Disabled");return;
    }
    bool Available=ReadFace();
    if (FaceHasBackup && Face && Face->GetAnimClass()!=URGBMoCapFaceAnimInstance::StaticClass()) {
        RestoreFace();FaceEnabled=false;FaceStatus=TEXT("Face animation changed externally - face disabled");return;
    }
    if (!Available && !FaceHasBackup) return;
    if (!Face || !Face->GetSkeletalMeshAsset()) {FaceStatus=TEXT("Target has no Face skeletal mesh");return;}
    if (!FaceHasBackup) {
        if (Face->GetAnimClass()==URGBMoCapFaceAnimInstance::StaticClass()) {FaceStatus=TEXT("Face preview has no recovery baseline. Disconnect before reconnecting.");return;}
        if (Face->GetAnimationMode()!=EAnimationMode::AnimationBlueprint) {FaceStatus=TEXT("Face must use Animation Blueprint mode");return;}
        if (!PrepareFaceMapping()) return;
        Face->Modify(false);BackedUpFace=Face;BackedUpFaceName=Face->GetFName();SavedFaceAnimClass=Face->GetAnimClass();SavedFaceMode=Face->GetAnimationMode();FaceHasBackup=true;
        SavedFaceVisibilityTick=Face->VisibilityBasedAnimTickOption;
#if WITH_EDITOR
        SavedFaceEditorTick=Face->GetUpdateAnimationInEditor();
#endif
        Face->SetAnimInstanceClass(URGBMoCapFaceAnimInstance::StaticClass());
        Face->VisibilityBasedAnimTickOption=EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;
#if WITH_EDITOR
        Face->SetUpdateAnimationInEditor(true);
#endif
    } else BackedUpFace=Face;
    MaintainPreviewTicks();
}
bool URGBMoCapConnectComponent::SampleFace(USkeletalMeshComponent* Mesh,TArray<FName>& Names,TArray<float>& Values) {
    if (!Enabled || !SupportsFace() || !FaceEnabled || !FaceHasBackup || Mesh!=FaceMesh(GetOwner()) || !PrepareFaceMapping()) return false;
    Names=FaceCurveNames;Values.Init(0.f,Names.Num());
    MapFace(FaceCurves,FaceMatrix,Values);
    // Body handles head/neck motion; facial ARKit input never adds head IK.
    Names.Add(TEXT("HeadControlSwitch"));Values.Add(0.f);return true;
}

void URGBMoCapConnectComponent::MapFace(TConstArrayView<float> Coefficients,TConstArrayView<float> Matrix,
                                        TArrayView<float> Values) {
    if (Coefficients.Num()!=52) return;
    for (int32 Channel=0;Channel<52;++Channel) {
        const float Weight=Coefficients[Channel];
        if (Weight==0.f) continue;
        const int32 Offset=Channel*Values.Num();
        for (int32 Curve=0;Curve<Values.Num();++Curve) Values[Curve]+=Weight*Matrix[Offset+Curve];
    }
}

#if WITH_EDITOR
void URGBMoCapConnectComponent::CaptureFaceBake(FRGBMoCapBakeContext& Context) {
    Context.FaceRequested=SupportsFace() && FaceEnabled;
    Context.FaceMesh.Reset();
    Context.FaceCurveNames.Reset();
    Context.FaceMatrix.Reset();
    Context.FaceWarning.Reset();
    if (!Context.FaceRequested) return;
    auto* Face=FaceMesh(GetOwner());
    auto* Mesh=Face?Face->GetSkeletalMeshAsset():nullptr;
    if (!Mesh || !Mesh->GetSkeleton()) {
        Context.FaceWarning=TEXT("Target has no Face mesh with a Skeleton asset.");return;
    }
    if (!PrepareFaceMapping()) {
        Context.FaceWarning=FaceStatus;return;
    }
    // Unmapped rig controls remain at their neutral default. Keep only the
    // controls the additive ARKit poses can actually drive.
    TArray<int32> DrivenCurves;
    const int32 CurveCount=FaceCurveNames.Num();
    for (int32 Curve=0;Curve<CurveCount;++Curve) {
        if (FaceCurveNames[Curve]==TEXT("HeadControlSwitch")) continue;
        for (int32 Channel=0;Channel<52;++Channel) {
            if (FaceMatrix[Channel*CurveCount+Curve]!=0.f) {
                DrivenCurves.Add(Curve);
                Context.FaceCurveNames.Add(FaceCurveNames[Curve]);
                break;
            }
        }
    }
    if (DrivenCurves.IsEmpty()) {
        Context.FaceWarning=TEXT("MetaHuman ARKit mapping drives no face controls.");return;
    }
    Context.FaceMatrix.Reserve(52*DrivenCurves.Num());
    for (int32 Channel=0;Channel<52;++Channel)
        for (int32 Curve:DrivenCurves) Context.FaceMatrix.Add(FaceMatrix[Channel*CurveCount+Curve]);
    Context.FaceMesh=Mesh;
}

bool URGBMoCapConnectComponent::MatchesFaceBakeTarget(const FRGBMoCapBakeContext& Context) const {
    auto* Face=FaceMesh(GetOwner());
    return SupportsFace() && FaceEnabled && Face && Face->GetSkeletalMeshAsset()==Context.FaceMesh.Get();
}
#endif
