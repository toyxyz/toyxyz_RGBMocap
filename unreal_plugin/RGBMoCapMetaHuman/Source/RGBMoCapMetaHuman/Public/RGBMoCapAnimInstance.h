#pragma once
#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "RGBMoCapAnimInstance.generated.h"

UCLASS(Transient)
class RGBMOCAPMETAHUMAN_API URGBMoCapAnimInstance : public UAnimInstance {
    GENERATED_BODY()
protected:
    virtual FAnimInstanceProxy* CreateAnimInstanceProxy() override;
    virtual void DestroyAnimInstanceProxy(FAnimInstanceProxy* Proxy) override;
};
