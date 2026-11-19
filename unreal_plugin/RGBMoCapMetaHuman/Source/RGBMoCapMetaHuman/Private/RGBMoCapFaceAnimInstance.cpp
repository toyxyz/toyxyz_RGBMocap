#line 1 "Source/RGBMoCapMetaHuman/Private/RGBMoCapFaceAnimInstance.cpp"
#include "RGBMoCapFaceAnimInstance.h"
#include "RGBMoCapConnectComponent.h"
#include "Animation/AnimInstanceProxy.h"
#include "Animation/AnimNodeBase.h"
#include "GameFramework/Actor.h"
class FRGBMoCapFaceProxy : public FAnimInstanceProxy {
    TArray<FName> Names;TArray<float> Values;
public:
    FRGBMoCapFaceProxy(UAnimInstance* Instance):FAnimInstanceProxy(Instance) {}
    virtual void PreUpdate(UAnimInstance* Instance,float Delta) override {
        FAnimInstanceProxy::PreUpdate(Instance,Delta);Names.Reset();Values.Reset();
        auto* Mesh=Instance->GetSkelMeshComponent();auto* Actor=Mesh?Mesh->GetOwner():nullptr;
        if (Actor) {
            TArray<URGBMoCapConnectComponent*> Connections;Actor->GetComponents(Connections);
            for (auto* C:Connections) if (C->SampleFace(Mesh,Names,Values)) break;
        }
    }
    virtual bool Evaluate(FPoseContext& Output) override {
        Output.ResetToRefPose();
        // The unchanged MetaHuman face postprocess copies body/head bones and
        // evaluates RigLogic from these mapped expression curves.
        for (int32 I=0;I<Names.Num();++I) Output.Curve.Set(Names[I],Values[I]);
        return true;
    }
};
FAnimInstanceProxy* URGBMoCapFaceAnimInstance::CreateAnimInstanceProxy() {return new FRGBMoCapFaceProxy(this);}
void URGBMoCapFaceAnimInstance::DestroyAnimInstanceProxy(FAnimInstanceProxy* Proxy) {delete Proxy;}
