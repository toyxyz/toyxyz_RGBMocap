#pragma once
#include "Commandlets/Commandlet.h"
#include "RGBMoCapLiveLinkCommandlet.generated.h"

UCLASS()
class URGBMoCapLiveLinkCommandlet : public UCommandlet {
    GENERATED_BODY()
public:
    URGBMoCapLiveLinkCommandlet();
    virtual int32 Main(const FString& Params) override;
};
