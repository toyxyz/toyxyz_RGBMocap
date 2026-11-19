#line 1 "Source/RGBMoCapMetaHuman/Private/RGBMoCapAnimInstance.cpp"
#include "RGBMoCapAnimInstance.h"
#include "RGBMoCapConnectComponent.h"
#include "Animation/AnimInstanceProxy.h"
#include "Animation/AnimNodeBase.h"
#include "GameFramework/Actor.h"

class FRGBMoCapProxy : public FAnimInstanceProxy {
    TArray<FTransform> Local;
public:
    FRGBMoCapProxy(UAnimInstance* Instance) : FAnimInstanceProxy(Instance) {}
    virtual void PreUpdate(UAnimInstance* Instance,float DeltaSeconds) override {
        FAnimInstanceProxy::PreUpdate(Instance,DeltaSeconds);
        auto* Mesh = Instance->GetSkelMeshComponent();
        auto* Owner = Mesh ? Mesh->GetOwner() : nullptr;
        bool Sampled=false;
        if (Owner) {
            TArray<URGBMoCapConnectComponent*> Connections;Owner->GetComponents(Connections);
            for (auto* C:Connections) if (C->SamplePose(Mesh,Local)) {Sampled=true;break;}
        }
        if (!Sampled) Local.Reset();
    }
    virtual bool Evaluate(FPoseContext& Output) override {
        Output.ResetToRefPose();
        const auto& Bones = Output.Pose.GetBoneContainer();
        for (FCompactPoseBoneIndex Index : Output.Pose.ForEachBoneIndex()) {
            int32 MeshIndex = Bones.MakeMeshPoseIndex(Index).GetInt();
            if (Local.IsValidIndex(MeshIndex)) Output.Pose[Index] = Local[MeshIndex];
        }
        return true;
    }
};
FAnimInstanceProxy* URGBMoCapAnimInstance::CreateAnimInstanceProxy() { return new FRGBMoCapProxy(this); }
void URGBMoCapAnimInstance::DestroyAnimInstanceProxy(FAnimInstanceProxy* Proxy) { delete Proxy; }
