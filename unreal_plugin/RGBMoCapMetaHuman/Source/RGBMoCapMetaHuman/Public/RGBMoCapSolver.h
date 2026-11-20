#pragma once
#include "CoreMinimal.h"

namespace RGBMoCap {
struct RGBMOCAPMETAHUMAN_API FSkeleton {
    TArray<FName> Names;
    TArray<int32> Parents;
    TArray<FTransform> Local;
    bool World(TArray<FTransform>& Result, FString& Error) const;
    int32 Find(const TCHAR* Name) const { return Names.IndexOfByKey(FName(Name)); }
};
struct RGBMOCAPMETAHUMAN_API FSettings {
    bool Ground = true;
    double FloorOffset = 0.;
};
struct RGBMOCAPMETAHUMAN_API FDiagnostics {
    double TravelScale = 1.;
    double PelvisAdjustment = 0.;
    double MinimumSole = 0.;
    double ReachError = 0.;
};
/** Independent geometric implementation; no Blender/Python code dependency. */
class RGBMOCAPMETAHUMAN_API FSolver {
public:
    bool Initialize(const FSkeleton& Source, const FSkeleton& Target, FString& Error, bool Mannequin=false);
    bool Evaluate(const TArray<FTransform>& SourceLocal, const FSettings& Settings,
                  TArray<FTransform>& Output, FDiagnostics& Diagnostics, FString& Error) const;
private:
    FSkeleton Source, Target;
    TArray<FTransform> SR, TR;
    TArray<int32> Mapping, AimSource, AimTarget, BlendSource;
    TArray<double> BlendWeight;
    TArray<FQuat> RotationReference;
    FQuat Alignment = FQuat::Identity;
    double Gain = 1.;
    int32 SP = INDEX_NONE, TP = INDEX_NONE;
    int32 SL[2][4]{}, TL[2][4]{};
    double Lengths[2][2]{};
    FVector Markers[2][2];
    bool UseMannequinGoals = false;
    FVector AnkleOffset = FVector::ZeroVector;
    TArray<TPair<int32,int32>> IKGuides;
};
}
