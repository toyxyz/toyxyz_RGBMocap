#pragma once

#include "CoreMinimal.h"
#include "RGBMoCapConnectComponent.h"

class UAnimSequence;

namespace RGBMoCap {
struct FBakedTracks {
    FFrameRate Rate;
    int32 Frames=0, Bones=0;
    TArray<FTransform3f> Keys;
    TArray<float> FaceKeys;
    FString FaceWarning;
};

bool SolveBake(const FRGBMoCapBakeContext& Context,const TArray<uint8>& Payload,FBakedTracks& Out,FString& Error);
UAnimSequence* SaveBake(const FRGBMoCapBakeContext& Context,const FBakedTracks& Animation,
                        const FString& ObjectPath,FString& Error,bool Face=false);
FString FaceBakePath(const FString& BodyPath);
bool CheckBakeDestination(const FString& ObjectPath,FString& Filename,FString& Error);
void RequestBake(FRGBMoCapBakeContext Context,TFunction<void(FBakedTracks,FString)> Complete);
}
