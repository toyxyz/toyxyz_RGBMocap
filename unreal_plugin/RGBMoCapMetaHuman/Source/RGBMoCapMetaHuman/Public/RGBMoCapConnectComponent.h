#pragma once
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Animation/AnimInstance.h"
#include "Components/SkeletalMeshComponent.h"
#include "RGBMoCapSolver.h"
#include "RGBMoCapConnectComponent.generated.h"

class UPoseAsset;

USTRUCT()
struct FRGBMoCapFollowerState {
    GENERATED_BODY()
    UPROPERTY() TObjectPtr<USkeletalMeshComponent> Mesh;
    UPROPERTY() FName ComponentName;
    UPROPERTY() bool EditorTick=false;
    UPROPERTY() EVisibilityBasedAnimTickOption VisibilityTick=EVisibilityBasedAnimTickOption::OnlyTickPoseWhenRendered;
};

UCLASS(ClassGroup=(Animation), meta=(BlueprintSpawnableComponent,DisplayName="RGB MoCap MetaHuman Connect"))
class RGBMOCAPMETAHUMAN_API URGBMoCapConnectComponent : public UActorComponent {
    GENERATED_BODY()
public:
    URGBMoCapConnectComponent();
    virtual ~URGBMoCapConnectComponent();
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RGB MoCap",meta=(GetOptions="GetSubjects"))
    FName Subject;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RGB MoCap")
    bool Enabled = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RGB MoCap")
    bool GroundContact = true;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RGB MoCap",meta=(Units="cm"))
    float FloorOffset = 0.f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RGB MoCap",AdvancedDisplay)
    FName BodyComponentName = TEXT("Body");
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Transient, Category="RGB MoCap")
    FString Status;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Transient, Category="RGB MoCap")
    float TravelScale = 1.f;
    UPROPERTY(EditAnywhere,BlueprintReadWrite,Category="RGB MoCap|Face") bool FaceEnabled=false;
    UPROPERTY(VisibleAnywhere,BlueprintReadOnly,Transient,Category="RGB MoCap|Face") FString FaceStatus=TEXT("Disabled");
    UPROPERTY(EditAnywhere,Category="RGB MoCap|Face",AdvancedDisplay) TSoftObjectPtr<UPoseAsset> FaceMappingPose;
    UFUNCTION(BlueprintCallable,Category="RGB MoCap|Face") void SetFaceEnabled(bool EnabledFace);
    bool SampleFace(USkeletalMeshComponent* Mesh,TArray<FName>& Names,TArray<float>& Values);
    UFUNCTION(BlueprintCallable,CallInEditor,Category="RGB MoCap") void Connect();
    UFUNCTION(BlueprintCallable,CallInEditor,Category="RGB MoCap") void Disconnect();
    UFUNCTION() TArray<FString> GetSubjects() const;
    UFUNCTION(BlueprintCallable,Category="RGB MoCap") static void FindAppSource();
    UFUNCTION(BlueprintCallable,Category="RGB MoCap") static void ReconnectAppSource();
    void SuspendForSave();
    void ResumeAfterSave();
    UFUNCTION(BlueprintCallable,Category="RGB MoCap")
    static URGBMoCapConnectComponent* ConnectActor(AActor* Actor,FName MotionSubject);
    bool SamplePose(USkeletalMeshComponent* Mesh, TArray<FTransform>& Pose);
    virtual void TickComponent(float DeltaTime,ELevelTick TickType,FActorComponentTickFunction* Tick) override;
    virtual void OnUnregister() override;
#if WITH_EDITOR
    virtual void PostEditUndo() override;
#endif
private:
    // Recovery identity/baseline survives duplication and saves but is not rolled
    // back with editable Subject/Enabled transactions. HasBackup is the active obligation.
    UPROPERTY(NonTransactional) TObjectPtr<USkeletalMeshComponent> BackedUpBody;
    UPROPERTY(NonTransactional) FName BackedUpBodyName;
    UPROPERTY(NonTransactional) TSubclassOf<UAnimInstance> SavedAnimClass;
    UPROPERTY(NonTransactional) TEnumAsByte<EAnimationMode::Type> SavedMode = EAnimationMode::AnimationBlueprint;
    UPROPERTY(NonTransactional) bool HasBackup = false;
    UPROPERTY(NonTransactional) bool SavedEditorTick = false;
    UPROPERTY(NonTransactional) EVisibilityBasedAnimTickOption SavedVisibilityTick = EVisibilityBasedAnimTickOption::OnlyTickPoseWhenRendered;
    UPROPERTY(NonTransactional) TArray<FRGBMoCapFollowerState> Followers;
    UPROPERTY(Transient,DuplicateTransient,NonTransactional) TObjectPtr<USkeletalMeshComponent> ActiveBody;
    TUniquePtr<RGBMoCap::FSolver> Solver;
    RGBMoCap::FSkeleton CachedSource;
    TWeakObjectPtr<USkeletalMesh> CachedMesh;
    FName CachedSubject;
    FGuid CachedSourceId;
    TArray<FTransform> CachedTargetRest;
    FVector CachedScale=FVector::OneVector;
    TArray<FTransform> LastPose;
    bool Applying = false;
    bool Saving = false;
    bool Reconcile = false;
    USkeletalMeshComponent* FindBody() const;
    UPROPERTY(NonTransactional) TObjectPtr<USkeletalMeshComponent> BackedUpFace;
    UPROPERTY(NonTransactional) FName BackedUpFaceName;
    UPROPERTY(NonTransactional) TSubclassOf<UAnimInstance> SavedFaceAnimClass;
    UPROPERTY(NonTransactional) TEnumAsByte<EAnimationMode::Type> SavedFaceMode=EAnimationMode::AnimationBlueprint;
    UPROPERTY(NonTransactional) bool FaceHasBackup=false;
    UPROPERTY(NonTransactional) bool SavedFaceEditorTick=false;
    UPROPERTY(NonTransactional) EVisibilityBasedAnimTickOption SavedFaceVisibilityTick=EVisibilityBasedAnimTickOption::OnlyTickPoseWhenRendered;
    UPROPERTY(Transient) TObjectPtr<UPoseAsset> MappingPose;
    TArray<FName> FaceCurveNames;
    TArray<float> FaceMatrix,FaceCurves;
    FName FaceSubject;FGuid FaceProvider;
    void UpdateFace();void RestoreFace();bool PrepareFaceMapping();bool ReadFace();
    void Restore();
    bool RepairReconstructedReferences();
    void MaintainPreviewTicks();
};
