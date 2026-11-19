#line 1 "Source/RGBMoCapMetaHuman/Private/RGBMoCapConnectComponent.cpp"
#include "RGBMoCapConnectComponent.h"
#include "RGBMoCapAnimInstance.h"
#include "Features/IModularFeatures.h"
#include "ILiveLinkClient.h"
#include "LiveLinkMessageBusSourceFactory.h"
#include "LiveLinkSourceSettings.h"
#include "Roles/LiveLinkAnimationRole.h"
#include "Roles/LiveLinkAnimationTypes.h"
#include "Engine/SkeletalMesh.h"
#include "GameFramework/Actor.h"
#include "HAL/PlatformTime.h"

namespace {
ILiveLinkClient* Client() {
    auto& Features=IModularFeatures::Get();
    return Features.IsModularFeatureAvailable(ILiveLinkClient::ModularFeatureName)
        ? &Features.GetModularFeature<ILiveLinkClient>(ILiveLinkClient::ModularFeatureName) : nullptr;
}
bool Read(const FLiveLinkSubjectFrameData& Frame,RGBMoCap::FSkeleton& Out) {
    const auto* S=Frame.StaticData.Cast<FLiveLinkSkeletonStaticData>();
    const auto* D=Frame.FrameData.Cast<FLiveLinkAnimationFrameData>();
    if (!S || !D) return false;
    Out.Names=S->BoneNames;Out.Parents=S->BoneParents;Out.Local=D->Transforms;
    return true;
}
bool Same(const RGBMoCap::FSkeleton& A,const RGBMoCap::FSkeleton& B) {
    if (A.Names!=B.Names || A.Parents!=B.Parents || A.Local.Num()!=B.Local.Num()) return false;
    for (int32 I=0;I<A.Local.Num();++I) if (!A.Local[I].Equals(B.Local[I],1.e-5)) return false;
    return true;
}
}
URGBMoCapConnectComponent::URGBMoCapConnectComponent() {
    PrimaryComponentTick.bCanEverTick=true;
    bTickInEditor=true;
    Status=TEXT("Choose a motion Subject and Connect.");
}
URGBMoCapConnectComponent::~URGBMoCapConnectComponent() = default;
USkeletalMeshComponent* URGBMoCapConnectComponent::FindBody() const {
    if (!GetOwner()) return nullptr;
    TArray<USkeletalMeshComponent*> Meshes;GetOwner()->GetComponents(Meshes);
    for (auto* Mesh:Meshes) if (Mesh->GetFName()==BodyComponentName) return Mesh;
    return nullptr;
}
TArray<FString> URGBMoCapConnectComponent::GetSubjects() const {
    TArray<FString> Result;
    if (auto* C=Client()) for (const auto& Key:C->GetSubjectsSupportingRole(ULiveLinkAnimationRole::StaticClass(),false,false)) {
        FString Name=Key.SubjectName.ToString();
        if (Name.StartsWith(TEXT("RGBMoCap_")) && !Name.EndsWith(TEXT("_Reference")) && !Name.EndsWith(TEXT("_Face"))) Result.AddUnique(Name);
    }
    Result.Sort();return Result;
}
URGBMoCapConnectComponent* URGBMoCapConnectComponent::ConnectActor(AActor* Actor,FName MotionSubject) {
    if (!Actor || Actor->IsTemplate()) return nullptr;
    Actor->Modify(false);
    auto* Component=Actor->FindComponentByClass<URGBMoCapConnectComponent>();
    if (!Component) {
        Component=NewObject<URGBMoCapConnectComponent>(Actor,NAME_None,RF_Transactional);
        Actor->AddInstanceComponent(Component);Component->RegisterComponent();
    }
    Component->Modify(false);Component->Subject=MotionSubject;Component->Connect();return Component;
}
void URGBMoCapConnectComponent::FindAppSource() {
    auto* C=Client();if (!C) return;
    for (const FGuid& Id:C->GetSources()) if (C->GetSourceType(Id).ToString()==TEXT("RGB MoCap")) return;
    FGuid Id=C->AddSource(GetDefault<ULiveLinkMessageBusSourceFactory>()->CreateSource(TEXT("Name=\"RGB MoCap\"")));
    if (auto* Settings=C->GetSourceSettings(Id)) Settings->Mode=ELiveLinkSourceMode::Latest;
}
void URGBMoCapConnectComponent::ReconnectAppSource() {
    auto* C=Client();if (!C) return;
    for (const FGuid& Id:C->GetSources()) if (C->GetSourceType(Id).ToString()==TEXT("RGB MoCap")) C->RemoveSource(Id);
    FindAppSource();
}
void URGBMoCapConnectComponent::SuspendForSave() { Saving=true;Restore(); }
void URGBMoCapConnectComponent::ResumeAfterSave() { Saving=false;if (Enabled) Connect(); }
void URGBMoCapConnectComponent::Connect() {
    if (Applying || Saving || IsTemplate()) return;
    if (GetOwner() && GetOwner()->FindComponentByClass<URGBMoCapConnectComponent>()!=this) {Status=TEXT("Use the existing RGB MoCap connection on this actor.");return;}
    TGuardValue<bool> Guard(Applying,true);
    auto* Body=FindBody();
    if (!Body || !Body->GetSkeletalMeshAsset()) { Status=TEXT("Select a MetaHuman actor with a Body skeletal mesh.");return; }
    if (Subject.IsNone() || Subject.ToString().EndsWith(TEXT("_Reference"))) { Status=TEXT("Choose the motion Subject without _Reference.");return; }
    const auto& Ref=Body->GetSkeletalMeshAsset()->GetRefSkeleton();
    for (const TCHAR* Name:{TEXT("pelvis"),TEXT("thigh_l"),TEXT("calf_l"),TEXT("foot_l"),TEXT("ball_l"),TEXT("thigh_r"),TEXT("calf_r"),TEXT("foot_r"),TEXT("ball_r")}) {
        if (Ref.FindBoneIndex(Name)==INDEX_NONE) { Status=FString::Printf(TEXT("Body is missing required MetaHuman bone: %s"),Name);return; }
    }
    if (Body->GetAnimationMode()!=EAnimationMode::AnimationBlueprint) { Status=TEXT("MetaHuman Body must use Animation Blueprint mode before connecting.");return; }
    FVector Scale=Body->GetComponentScale();
    if (!Scale.AllComponentsEqual(1.e-4) || Scale.X<=0.) { Status=TEXT("Use positive uniform scale on the MetaHuman Body.");return; }
    if (HasBackup && BackedUpBody!=Body) Restore();
    if (Body->GetAnimClass()==URGBMoCapAnimInstance::StaticClass() && !(HasBackup && BackedUpBody==Body)) { Status=TEXT("Another connection owns this Body. Disconnect it first.");return; }
    if (ActiveBody && ActiveBody!=Body) Restore();
    ActiveBody=Body;
    if (Body->GetAnimClass()!=URGBMoCapAnimInstance::StaticClass()) {
        Body->Modify(false);Followers.Reset();
        BackedUpBody=Body;BackedUpBodyName=Body->GetFName();SavedAnimClass=Body->GetAnimClass();SavedMode=Body->GetAnimationMode();HasBackup=true;
        SavedVisibilityTick=Body->VisibilityBasedAnimTickOption;
#if WITH_EDITOR
        SavedEditorTick=Body->GetUpdateAnimationInEditor();
#endif
        Body->SetAnimInstanceClass(URGBMoCapAnimInstance::StaticClass());
    }
    if (Followers.IsEmpty()) {
        TArray<USkeletalMeshComponent*> Meshes;GetOwner()->GetComponents(Meshes);
        for (auto* Mesh:Meshes) if (Mesh!=Body && Mesh->IsAttachedTo(Body)) {
            FRGBMoCapFollowerState F;F.Mesh=Mesh;F.ComponentName=Mesh->GetFName();F.VisibilityTick=Mesh->VisibilityBasedAnimTickOption;
#if WITH_EDITOR
            F.EditorTick=Mesh->GetUpdateAnimationInEditor();
#endif
            Followers.Add(F);
        }
    }
    for (const auto& F:Followers) if (F.Mesh) {
        if (F.Mesh->VisibilityBasedAnimTickOption!=EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones
#if WITH_EDITOR
            || !F.Mesh->GetUpdateAnimationInEditor()
#endif
        ) F.Mesh->Modify(false);
        F.Mesh->VisibilityBasedAnimTickOption=EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;
#if WITH_EDITOR
        F.Mesh->SetUpdateAnimationInEditor(true);
#endif
    }
    Body->VisibilityBasedAnimTickOption=EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;
#if WITH_EDITOR
    Body->SetUpdateAnimationInEditor(true);
#endif
    Enabled=true;FindAppSource();Status=TEXT("Connected. Waiting for motion and matching reference.");
}
void URGBMoCapConnectComponent::MaintainPreviewTicks() {
    // MetaHuman's On Anim Initialized (Face) event can reset editor-update
    // flags on EVERY skeletal component. Reapply only our owned preview flags
    // after that callback, without recapturing or changing recovery baselines.
    if (!Enabled || Saving || !HasBackup || !IsValid(ActiveBody)
        || ActiveBody->GetAnimClass()!=URGBMoCapAnimInstance::StaticClass()) return;
    auto Apply=[](USkeletalMeshComponent* Mesh) {
        if (!IsValid(Mesh)) return;
        if (Mesh->VisibilityBasedAnimTickOption!=EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones
#if WITH_EDITOR
            || !Mesh->GetUpdateAnimationInEditor()
#endif
        ) Mesh->Modify(false);
        Mesh->VisibilityBasedAnimTickOption=EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;
#if WITH_EDITOR
        Mesh->SetUpdateAnimationInEditor(true);
#endif
    };
    Apply(ActiveBody);
    for (const auto& F:Followers) Apply(F.Mesh);
}
bool URGBMoCapConnectComponent::RepairReconstructedReferences() {
    if (!GetOwner() || BackedUpBodyName.IsNone()) return false;
    TArray<USkeletalMeshComponent*> Meshes;GetOwner()->GetComponents(Meshes);
    bool Replaced=false;
    if (!IsValid(BackedUpBody) || !Meshes.Contains(BackedUpBody.Get()) || BackedUpBody->GetFName()!=BackedUpBodyName) {
        for (auto* M:Meshes) if (M->GetFName()==BackedUpBodyName) {BackedUpBody=M;Replaced=true;break;}
    }
    for (auto& F:Followers) if (!F.ComponentName.IsNone() && (!IsValid(F.Mesh) || !Meshes.Contains(F.Mesh.Get()) || F.Mesh->GetFName()!=F.ComponentName)) {
        for (auto* M:Meshes) if (M->GetFName()==F.ComponentName) {F.Mesh=M;break;}
    }
    return Replaced;
}
void URGBMoCapConnectComponent::Restore() {
    RestoreFace();
    bool Replaced=RepairReconstructedReferences();
    if (Replaced && (Enabled || Reconcile || Saving) && IsValid(BackedUpBody) && BackedUpBody->GetAnimClass()==URGBMoCapAnimInstance::StaticClass()) HasBackup=true;
    // This identity is remapped by actor duplication; the transient binding may
    // not exist yet, and BodyComponentName may have changed since connection.
    auto* SavedBody=IsValid(BackedUpBody)?BackedUpBody.Get():nullptr;
    if (SavedBody && HasBackup) {
        SavedBody->Modify(false);
        if (SavedBody->GetAnimClass()==URGBMoCapAnimInstance::StaticClass()) {
            SavedBody->SetAnimInstanceClass(SavedAnimClass);SavedBody->SetAnimationMode(SavedMode);
        }
        if (SavedBody->VisibilityBasedAnimTickOption==EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones) SavedBody->VisibilityBasedAnimTickOption=SavedVisibilityTick;
#if WITH_EDITOR
        SavedBody->SetUpdateAnimationInEditor(SavedEditorTick);
#endif
    }
    if (HasBackup) for (const auto& F:Followers) if (IsValid(F.Mesh)) {
        F.Mesh->Modify(false);
        if (F.Mesh->VisibilityBasedAnimTickOption==EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones) F.Mesh->VisibilityBasedAnimTickOption=F.VisibilityTick;
#if WITH_EDITOR
        F.Mesh->SetUpdateAnimationInEditor(F.EditorTick);
#endif
    }
    // Retain a nontransactional baseline only for deliberate undo reconciliation.
    // It does not authorize normal adoption of an unowned native AnimClass.
    HasBackup=false;
    ActiveBody=nullptr;Solver.Reset();CachedMesh.Reset();CachedSubject=NAME_None;CachedSourceId.Invalidate();CachedTargetRest.Reset();LastPose.Reset();
}
void URGBMoCapConnectComponent::Disconnect() {
    Restore();Enabled=false;HasBackup=false;Status=TEXT("Disconnected. Original animation restored.");
}
void URGBMoCapConnectComponent::OnUnregister() { Restore();Super::OnUnregister(); }
#if WITH_EDITOR
void URGBMoCapConnectComponent::PostEditUndo() {
    Super::PostEditUndo();ActiveBody=nullptr;Solver.Reset();LastPose.Reset();Reconcile=true;
}
#endif
void URGBMoCapConnectComponent::TickComponent(float DeltaTime,ELevelTick TickType,FActorComponentTickFunction* Tick) {
    Super::TickComponent(DeltaTime,TickType,Tick);
    UpdateFace();
    bool Replaced=RepairReconstructedReferences();
    if (Replaced && Enabled && IsValid(BackedUpBody) && BackedUpBody->GetAnimClass()==URGBMoCapAnimInstance::StaticClass()) HasBackup=true;
    if (Reconcile) {
        Reconcile=false;auto* Body=FindBody();
        if (!HasBackup && Body && BackedUpBody==Body && Body->GetAnimClass()==URGBMoCapAnimInstance::StaticClass()) HasBackup=true;
        if (!Enabled) Restore();
    }
    if (ActiveBody && ActiveBody!=FindBody()) Restore();
    if (Enabled && !Saving && !ActiveBody) Connect();
    if (!Enabled && (ActiveBody || HasBackup)) Restore();
    if (ActiveBody && ActiveBody->GetAnimClass()!=URGBMoCapAnimInstance::StaticClass()) {
        Enabled=false;Restore();Status=TEXT("Animation changed externally. Connection released without replacing it.");
    }
    MaintainPreviewTicks();
}
bool URGBMoCapConnectComponent::SamplePose(USkeletalMeshComponent* Mesh,TArray<FTransform>& Pose) {
    if (!Enabled || Mesh!=ActiveBody) return false;
    auto* Asset=Mesh->GetSkeletalMeshAsset();
    FVector Scale=Mesh->GetComponentScale();
    if (!Asset || !Scale.AllComponentsEqual(1.e-4) || Scale.X<=0.) { Status=TEXT("Unsupported Body mesh or nonuniform scale.");LastPose.Reset();return false; }
    const auto& TargetRest=Asset->GetRefSkeleton().GetRefBonePose();
    bool TargetChanged=CachedTargetRest.Num()!=TargetRest.Num();
    for (int32 I=0;!TargetChanged && I<TargetRest.Num();++I) TargetChanged=!CachedTargetRest[I].Equals(TargetRest[I],1.e-5);
    if (CachedSubject!=Subject || CachedMesh!=Asset || !CachedScale.Equals(Scale,1.e-5) || TargetChanged) { Solver.Reset();LastPose.Reset(); }
    auto* C=Client();FLiveLinkSubjectFrameData Reference,Motion;
    FGuid MotionSource,ReferenceSource;int32 MotionMatches=0,ReferenceMatches=0;
    if (C) for (const auto& Key:C->GetSubjectsSupportingRole(ULiveLinkAnimationRole::StaticClass(),false,false)) {
        if (!C->IsSubjectEnabled(Key,true)) continue;
        if (Key.SubjectName==Subject) {MotionSource=Key.Source;++MotionMatches;}
        if (Key.SubjectName==FName(*(Subject.ToString()+TEXT("_Reference")))) {ReferenceSource=Key.Source;++ReferenceMatches;}
    }
    if (MotionMatches!=1 || ReferenceMatches!=1 || MotionSource!=ReferenceSource) {
        Status=TEXT("Waiting for one matching motion/reference pair from the same provider.");Pose=LastPose;return !Pose.IsEmpty();
    }
    if (CachedSourceId.IsValid() && CachedSourceId!=MotionSource) {Solver.Reset();LastPose.Reset();}
    if (!C || !C->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(MotionSource,Subject),ULiveLinkAnimationRole::StaticClass(),Motion)
        || !C->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(ReferenceSource,FName(*(Subject.ToString()+TEXT("_Reference")))),ULiveLinkAnimationRole::StaticClass(),Reference)) {
        Status=TEXT("Waiting for motion/reference. Check the app and Live Link source.");Pose=LastPose;return !Pose.IsEmpty();
    }
    const auto* Frame=Motion.FrameData.Cast<FLiveLinkAnimationFrameData>();
    const auto* RefFrame=Reference.FrameData.Cast<FLiveLinkAnimationFrameData>();
    auto Age=[](const FLiveLinkAnimationFrameData* F) {return F?FPlatformTime::Seconds()-(F->ArrivalTime.WorldTime>0.?F->ArrivalTime.WorldTime:F->WorldTime.GetOffsettedTime()):DBL_MAX;};
    if (!Frame || !RefFrame || !FMath::IsFinite(Age(Frame)) || !FMath::IsFinite(Age(RefFrame)) || Age(Frame)<-1. || Age(RefFrame)<-1. || Age(Frame)>2. || Age(RefFrame)>2.) {
        Status=TEXT("Stream is stale; holding last pose. Reconnect the Live Link source if needed.");Pose=LastPose;return !Pose.IsEmpty();
    }
    const FString* MotionRig=Frame->MetaData.StringMetaData.Find(TEXT("RigId"));
    const FString* ReferenceRig=RefFrame->MetaData.StringMetaData.Find(TEXT("RigId"));
    if (!MotionRig || !ReferenceRig || MotionRig->IsEmpty() || *MotionRig!=*ReferenceRig) {
        Status=TEXT("Waiting for matching motion/reference rig revisions.");Pose=LastPose;return !Pose.IsEmpty();
    }
    RGBMoCap::FSkeleton S,M;
    if (!Read(Reference,S) || !Read(Motion,M) || S.Names!=M.Names || S.Parents!=M.Parents) {
        Status=TEXT("Waiting for matching motion/reference skeletons.");Solver.Reset();LastPose.Reset();return false;
    }
    if (!Solver || CachedMesh!=Asset || CachedSubject!=Subject || !Same(CachedSource,S)) {
        Solver.Reset();LastPose.Reset();
        RGBMoCap::FSkeleton T;const auto& Ref=Asset->GetRefSkeleton();
        T.Local=Ref.GetRefBonePose();
        for (int32 I=0;I<Ref.GetNum();++I) { T.Names.Add(Ref.GetBoneName(I));T.Parents.Add(Ref.GetParentIndex(I)); }
        auto NewSolver=MakeUnique<RGBMoCap::FSolver>();
        if (!NewSolver->Initialize(S,T,Status)) return false;
        Solver=MoveTemp(NewSolver);CachedSource=MoveTemp(S);CachedMesh=Asset;CachedSubject=Subject;
        CachedSourceId=MotionSource;CachedTargetRest=TargetRest;CachedScale=Scale;
    }
    if (!FMath::IsFinite(FloorOffset)) {Status=TEXT("Floor Offset must be finite.");Pose=LastPose;return !Pose.IsEmpty();}
    RGBMoCap::FSettings Options;Options.Ground=GroundContact;Options.FloorOffset=FloorOffset;
    RGBMoCap::FDiagnostics Diagnostics;
    if (!Solver->Evaluate(M.Local,Options,Pose,Diagnostics,Status)) {Pose=LastPose;return !Pose.IsEmpty();}
    LastPose=Pose;TravelScale=Diagnostics.TravelScale;
    Status=Diagnostics.ReachError>.5?TEXT("Streaming - foot reach limited by target proportions."):
        TEXT("Streaming - direct body retarget; flat-floor sole geometry is estimated.");return true;
}
