#pragma once
#include "Animation/AnimInstance.h"
#include "RGBMoCapFaceAnimInstance.generated.h"
UCLASS(Transient)
class RGBMOCAPMETAHUMAN_API URGBMoCapFaceAnimInstance : public UAnimInstance {
    GENERATED_BODY()
protected:
    virtual FAnimInstanceProxy* CreateAnimInstanceProxy() override;
    virtual void DestroyAnimInstanceProxy(FAnimInstanceProxy* Proxy) override;
};
