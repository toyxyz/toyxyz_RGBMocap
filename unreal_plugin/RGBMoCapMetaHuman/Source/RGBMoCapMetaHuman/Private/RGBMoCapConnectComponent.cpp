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
#include "GameFramework/Character.h"
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
bool ReadBodyFrames(FName Subject,FGuid& MotionSource,FLiveLinkSubjectFrameData& Motion,
                    FLiveLinkSubjectFrameData& Reference,FString& Error) {
    auto* C=Client();
    FGuid ReferenceSource;
    int32 MotionMatches=0,ReferenceMatches=0;
    const FName RefName(*(Subject.ToString()+TEXT("_Reference")));
    if (C) for (const auto& Key:C->GetSubjectsSupportingRole(ULiveLinkAnimationRole::StaticClass(),false,false)) {
        if (!C->IsSubjectEnabled(Key,true)) continue;
        if (Key.SubjectName==Subject) {MotionSource=Key.Source;++MotionMatches;}
        if (Key.SubjectName==RefName) {ReferenceSource=Key.Source;++ReferenceMatches;}
    }
    if (!C || MotionMatches!=1 || ReferenceMatches!=1 || MotionSource!=ReferenceSource) {
        MotionSource.Invalidate();
        Error=TEXT("Waiting for one matching motion/reference pair from the same provider.");
        return false;
    }
    const auto Role=ULiveLinkAnimationRole::StaticClass();
    if (!C->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(MotionSource,Subject),Role,Motion)
        || !C->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(ReferenceSource,RefName),Role,Reference)) {
        Error=TEXT("Waiting for motion/reference. Check the app and Live Link source.");
        return false;
    }
    const auto* Frame=Motion.FrameData.Cast<FLiveLinkAnimationFrameData>();
    const auto* RefFrame=Reference.FrameData.Cast<FLiveLinkAnimationFrameData>();
    const auto Fresh=[](const FLiveLinkAnimationFrameData* F) {
        if (!F) return false;
        const double Stamp=F->ArrivalTime.WorldTime>0.?F->ArrivalTime.WorldTime:F->WorldTime.GetOffsettedTime();
        const double Age=FPlatformTime::Seconds()-Stamp;
        return FMath::IsFinite(Age) && Age>=-1. && Age<=2.;
    };
    if (!Fresh(Frame) || !Fresh(RefFrame)) {
        Error=TEXT("Stream is stale; holding last pose. Reconnect the Live Link source if needed.");
        return false;
    }
    const FString* A=Frame->MetaData.StringMetaData.Find(TEXT("RigId"));
    const FString* B=RefFrame->MetaData.StringMetaData.Find(TEXT("RigId"));
    if (!A || !B || A->IsEmpty() || *A!=*B) {
        Error=TEXT("Waiting for matching motion/reference rig revisions.");
        return false;
    }
    return true;
}
RGBMoCap::FSkeleton TargetSkeleton(const FReferenceSkeleton& Ref) {
    RGBMoCap::FSkeleton Out;
    Out.Local=Ref.GetRefBonePose();
    for (int32 I=0;I<Ref.GetNum();++I) {Out.Names.Add(Ref.GetBoneName(I));Out.Parents.Add(Ref.GetParentIndex(I));}
    return Out;
}
bool Same(const RGBMoCap::FSkeleton& A,const RGBMoCap::FSkeleton& B) {
    if (A.Names!=B.Names || A.Parents!=B.Parents || A.Local.Num()!=B.Local.Num()) return false;
    for (int32 I=0;I<A.Local.Num();++I) if (!A.Local[I].Equals(B.Local[I],1.e-5)) return false;
    return true;
}
USkeletalMeshComponent* NamedMesh(const AActor* Actor,FName Name) {
    if (!Actor || Name.IsNone()) return nullptr;
    TArray<USkeletalMeshComponent*> Meshes;
    Actor->GetComponents(Meshes);
    for (auto* Mesh:Meshes) if (IsValid(Mesh) && Mesh->GetFName()==Name) return Mesh;
    return nullptr;
}
bool SupportedBody(const FReferenceSkeleton& Ref) {
    for (const TCHAR* Name:{TEXT("pelvis"),TEXT("thigh_l"),TEXT("calf_l"),TEXT("foot_l"),TEXT("ball_l"),
        TEXT("thigh_r"),TEXT("calf_r"),TEXT("foot_r"),TEXT("ball_r")})
        if (Ref.FindBoneIndex(Name)==INDEX_NONE) return false;
    return true;
}
bool IsMannequinMesh(const USkeletalMeshComponent* Mesh) {
    const USkeletalMesh* Asset=IsValid(Mesh)?Mesh->GetSkeletalMeshAsset():nullptr;
    if (!Asset) return false;
    const FReferenceSkeleton& Ref=Asset->GetRefSkeleton();
    if (Ref.FindBoneIndex(TEXT("root"))!=0 || !SupportedBody(Ref)) return false;
    for (const TCHAR* Name:{TEXT("spine_05"),TEXT("head"),TEXT("hand_l"),TEXT("hand_r"),TEXT("ik_foot_root"),
        TEXT("ik_foot_l"),TEXT("ik_foot_r"),TEXT("ik_hand_root"),TEXT("ik_hand_gun"),TEXT("ik_hand_l"),TEXT("ik_hand_r")})
        if (Ref.FindBoneIndex(Name)==INDEX_NONE) return false;
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
    return NamedMesh(GetOwner(),TargetType==ERGBMoCapTargetType::Mannequin?BackedUpBodyName:BodyComponentName);
}
USkeletalMeshComponent* URGBMoCapConnectComponent::FindTargetMesh(const AActor* Actor,ERGBMoCapTargetType Type,FString& Reason) {
    Reason.Reset();
    if (!IsValid(Actor) || Actor->IsTemplate()) {Reason=TEXT("Select a target actor in the level.");return nullptr;}
    const auto* Con=Actor->FindComponentByClass<URGBMoCapConnectComponent>();
    auto* Face=NamedMesh(Actor,TEXT("Face"));
    const bool HasFace=Face && Face->GetSkeletalMeshAsset();
    USkeletalMeshComponent* Mesh=nullptr;
    if (Type==ERGBMoCapTargetType::MetaHuman) {
        Mesh=NamedMesh(Actor,Con?Con->BodyComponentName:FName(TEXT("Body")));
        if (!Mesh || !Mesh->GetSkeletalMeshAsset() || (!HasFace && IsMannequinMesh(Mesh))) {
            Reason=TEXT("Select a MetaHuman actor with a Body skeletal mesh.");return nullptr;
        }
        if (!SupportedBody(Mesh->GetSkeletalMeshAsset()->GetRefSkeleton())) {
            Reason=TEXT("The MetaHuman Body is missing required pelvis or leg bones.");return nullptr;
        }
    } else if (Type==ERGBMoCapTargetType::Mannequin) {
        if (HasFace && NamedMesh(Actor,TEXT("Body"))) {
            Reason=TEXT("Select MetaHuman for an actor with Body and Face meshes.");return nullptr;
        }
        if (const auto* Character=Cast<ACharacter>(Actor); Character && IsMannequinMesh(Character->GetMesh()))
            Mesh=Character->GetMesh();
        if (!Mesh) {
            TArray<USkeletalMeshComponent*> Meshes;
            Actor->GetComponents(Meshes);
            for (auto* Candidate:Meshes) if (IsMannequinMesh(Candidate)) {
                if (Mesh) {Reason=TEXT("The actor has multiple mannequin meshes. Select an actor with one target mesh.");return nullptr;}
                Mesh=Candidate;
            }
        }
        if (!Mesh) {Reason=TEXT("Select an actor with a UE5 Manny or Quinn skeletal mesh.");return nullptr;}
    } else {
        Reason=TEXT("Select a supported target type.");return nullptr;
    }
    const FVector Scale=Mesh->GetComponentScale();
    if (!Scale.AllComponentsEqual(1.e-4) || Scale.X<=0.) {
        Reason=TEXT("The target mesh needs positive uniform scale.");return nullptr;
    }
    if (Mesh->GetAnimationMode()!=EAnimationMode::AnimationBlueprint) {
        Reason=TEXT("Set the target mesh to Use Animation Blueprint mode.");return nullptr;
    }
    return Mesh;
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
    return ConnectActorForTarget(Actor,MotionSubject,ERGBMoCapTargetType::MetaHuman);
}
URGBMoCapConnectComponent* URGBMoCapConnectComponent::ConnectActorForTarget(AActor* Actor,FName MotionSubject,ERGBMoCapTargetType Type) {
    FString Reason;
    auto* Mesh=FindTargetMesh(Actor,Type,Reason);
    if (!Mesh || MotionSubject.IsNone() || MotionSubject.ToString().EndsWith(TEXT("_Reference"))
        || MotionSubject.ToString().EndsWith(TEXT("_Face"))) return nullptr;
    auto* Component=Actor->FindComponentByClass<URGBMoCapConnectComponent>();
    if (Mesh->GetAnimClass()==URGBMoCapAnimInstance::StaticClass() &&
        (!Component || !Component->HasBackup || Component->BackedUpBody!=Mesh)) return nullptr;
    Actor->Modify(false);
    if (!Component) {
        Component=NewObject<URGBMoCapConnectComponent>(Actor,NAME_None,RF_Transactional);
        Actor->AddInstanceComponent(Component);Component->RegisterComponent();
    }
    Component->Modify(false);
    if (Component->TargetType!=Type) Component->Restore();
    Component->TargetType=Type;
    Component->Subject=MotionSubject;
    Component->Connect();
    return Component;
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
    auto* Body=FindTargetMesh(GetOwner(),TargetType,Status);
    if (!Body) return;
    if (Subject.IsNone() || Subject.ToString().EndsWith(TEXT("_Reference")) || Subject.ToString().EndsWith(TEXT("_Face"))) {
        Status=TEXT("Choose the body motion Subject.");return;
    }
    if (HasBackup && BackedUpBody!=Body) Restore();
    if (Body->GetAnimClass()==URGBMoCapAnimInstance::StaticClass() && !(HasBackup && BackedUpBody==Body)) { Status=TEXT("Another connection owns this Body. Disconnect it first.");return; }
    if (ActiveBody && ActiveBody!=Body) Restore();
    ActiveBody=Body;
    ActiveTargetType=TargetType;
    if (!SupportsFace()) SetFaceEnabled(false);
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
    if (ActiveBody && ActiveTargetType!=TargetType) Restore();
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
    if (!Enabled || Mesh!=ActiveBody || ActiveTargetType!=TargetType) return false;
    auto* Asset=Mesh->GetSkeletalMeshAsset();
    FVector Scale=Mesh->GetComponentScale();
    if (!Asset || !Scale.AllComponentsEqual(1.e-4) || Scale.X<=0.) { Status=TEXT("Unsupported Body mesh or nonuniform scale.");LastPose.Reset();return false; }
    if (TargetType==ERGBMoCapTargetType::Mannequin && CachedMesh!=Asset && !IsMannequinMesh(Mesh)) {
        Status=TEXT("The target mesh no longer has the UE5 mannequin rig.");LastPose.Reset();return false;
    }
    const auto& TargetRest=Asset->GetRefSkeleton().GetRefBonePose();
    bool TargetChanged=CachedTargetRest.Num()!=TargetRest.Num();
    for (int32 I=0;!TargetChanged && I<TargetRest.Num();++I) TargetChanged=!CachedTargetRest[I].Equals(TargetRest[I],1.e-5);
    if (CachedSubject!=Subject || CachedMesh!=Asset || !CachedScale.Equals(Scale,1.e-5) || TargetChanged) { Solver.Reset();LastPose.Reset(); }
    FLiveLinkSubjectFrameData Reference,Motion;
    FGuid MotionSource;
    const bool Ready=ReadBodyFrames(Subject,MotionSource,Motion,Reference,Status);
    if (MotionSource.IsValid() && CachedSourceId.IsValid() && CachedSourceId!=MotionSource) {
        Solver.Reset();LastPose.Reset();
    }
    if (!Ready) {Pose=LastPose;return !Pose.IsEmpty();}
    RGBMoCap::FSkeleton S,M;
    if (!Read(Reference,S) || !Read(Motion,M) || S.Names!=M.Names || S.Parents!=M.Parents) {
        Status=TEXT("Waiting for matching motion/reference skeletons.");Solver.Reset();LastPose.Reset();return false;
    }
    if (!Solver || CachedMesh!=Asset || CachedSubject!=Subject || !Same(CachedSource,S)) {
        Solver.Reset();LastPose.Reset();
        const auto T=TargetSkeleton(Asset->GetRefSkeleton());
        auto NewSolver=MakeUnique<RGBMoCap::FSolver>();
        if (!NewSolver->Initialize(S,T,Status,TargetType==ERGBMoCapTargetType::Mannequin)) return false;
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


#if WITH_EDITOR
bool URGBMoCapConnectComponent::MatchesBakeTarget(const FRGBMoCapBakeContext& Context) const {
    return Enabled && !Saving && IsValid(ActiveBody) && ActiveTargetType==TargetType && Subject==Context.Subject
        && ActiveBody->GetSkeletalMeshAsset()==Context.Mesh.Get()
        && (TargetType==ERGBMoCapTargetType::Mannequin)==Context.Mannequin;
}

bool URGBMoCapConnectComponent::CaptureBake(FRGBMoCapBakeContext& Context,FString& Error) {
    if (!Enabled || Saving || !IsValid(ActiveBody) || TargetType!=ActiveTargetType) {
        Error=TEXT("Connect a target before baking.");return false;
    }
    auto* Mesh=ActiveBody->GetSkeletalMeshAsset();
    const auto Scale=ActiveBody->GetComponentScale();
    if (!Mesh || !Mesh->GetSkeleton() || !Scale.AllComponentsEqual(1.e-4) || Scale.X<=0. || !FMath::IsFinite(FloorOffset)) {
        Error=TEXT("The target needs a skeletal mesh with uniform positive scale and a finite floor offset.");return false;
    }
    FLiveLinkSubjectFrameData Motion,Reference;
    FGuid Provider;
    if (!ReadBodyFrames(Subject,Provider,Motion,Reference,Error)) return false;
    RGBMoCap::FSkeleton Current;
    if (!Read(Reference,Context.Source) || !Read(Motion,Current) || Context.Source.Names!=Current.Names
        || Context.Source.Parents!=Current.Parents) {
        Error=TEXT("Waiting for matching motion/reference skeletons.");return false;
    }
    const auto& Meta=Motion.FrameData.Cast<FLiveLinkAnimationFrameData>()->MetaData.StringMetaData;
    const auto* Port=Meta.Find(TEXT("RGBBakePort"));
    const auto* Token=Meta.Find(TEXT("RGBBakeToken"));
    const auto* Take=Meta.Find(TEXT("RGBBakeTake"));
    const auto IsHex=[](const FString* S) {
        if (!S || S->Len()!=64) return false;
        for (TCHAR Ch:*S) if (!FChar::IsHexDigit(Ch)) return false;
        return true;
    };
    int32 Number=0;
    if (!Port || !LexTryParseString(Number,**Port) || Number<1 || Number>65535
        || !IsHex(Token) || !IsHex(Take)) {
        Error=TEXT("Bake needs an updated RGB MoCap app and sender running on this PC.");return false;
    }
    Context.Subject=Subject;
    Context.Rig=Meta.FindChecked(TEXT("RigId"));
    Context.Take=*Take;
    Context.Token=*Token;
    Context.Port=Number;
    Context.Mesh=Mesh;
    Context.RawBones=Mesh->GetRefSkeleton().GetRawBoneNum();
    Context.Target=TargetSkeleton(Mesh->GetRefSkeleton());
    Context.Mannequin=TargetType==ERGBMoCapTargetType::Mannequin;
    Context.Settings.Ground=GroundContact;
    Context.Settings.FloorOffset=FloorOffset;
    CaptureFaceBake(Context);
    return true;
}
#endif
