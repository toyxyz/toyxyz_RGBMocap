#line 1 "Source/RGBMoCapMetaHuman/Private/RGBMoCapSolver.cpp"
#include "RGBMoCapSolver.h"
#include "Math/RotationMatrix.h"

namespace RGBMoCap {
namespace {
bool Finite(const FTransform& T) {
    return !T.ContainsNaN() && T.GetRotation().IsNormalized()
        && T.GetScale3D().Equals(FVector::OneVector,1.e-4);
}
FQuat Basis(const FVector& Right,const FVector& Up) {
    // Both contracts are Z-up. Anatomical lean must never tilt planar locomotion.
    FVector Z=FVector::UpVector,Y=Right;Y.Z=0.;Y.Normalize();
    return FRotationMatrix::MakeFromYZ(Y,Z).ToQuat();
}
FQuat ClavicleRestLift(const FVector& SourceClavicle,const FVector& SourceArm,
                      const FVector& TargetClavicle,const FVector& TargetArm) {
    if (SourceClavicle.IsNearlyZero() || SourceArm.IsNearlyZero() || TargetClavicle.IsNearlyZero() || TargetArm.IsNearlyZero())
        return FQuat::Identity;
    const auto Elevation=[](const FVector& V) { return FMath::Atan2(V.Z,V.Size2D()); };
    const FVector Axis=FVector::CrossProduct(TargetClavicle,FVector::UpVector).GetSafeNormal();
    if (Axis.IsNearlyZero()) return FQuat::Identity;
    // Match the offline tolerance and horizontal-projection ramp.
    const double Drop=FMath::Clamp(-Elevation(TargetArm)-FMath::Max(FMath::DegreesToRadians(10.),-Elevation(SourceArm)),0.,UE_HALF_PI);
    const double Gap=FMath::Max(0.,Elevation(SourceClavicle)-Elevation(TargetClavicle));
    return FQuat(Axis,(1.-FMath::Cos(Drop))*Gap);
}
void Compose(const FSkeleton& Skeleton,const TArray<FTransform>& Local,TArray<FTransform>& World) {
    World.SetNum(Local.Num());
    for (int32 I=0;I<Local.Num();++I) World[I]=Skeleton.Parents[I]<0?Local[I]:Local[I]*World[Skeleton.Parents[I]];
}
void SetWorldRotation(const FSkeleton& S,int32 I,FQuat Q,TArray<FTransform>& Local,TArray<FTransform>& World) {
    Q.Normalize();int32 P=S.Parents[I];
    Local[I].SetRotation(P<0?Q:World[P].GetRotation().Inverse()*Q);
    Compose(S,Local,World);
}
void SetWorldPosition(const FSkeleton& S,int32 I,FVector P,TArray<FTransform>& Local,TArray<FTransform>& World) {
    int32 Parent=S.Parents[I];
    Local[I].SetTranslation(Parent<0?P:World[Parent].InverseTransformPosition(P));
    Compose(S,Local,World);
}
}
bool FSkeleton::World(TArray<FTransform>& Result,FString& Error) const {
    int32 N=Names.Num();
    if (N<1 || N>2048 || Parents.Num()!=N || Local.Num()!=N) { Error=TEXT("Invalid skeleton array sizes.");return false; }
    TSet<FName> Seen;
    for (int32 I=0;I<N;++I) {
        if (Names[I].IsNone() || Seen.Contains(Names[I]) || (I==0?Parents[I]!=-1:Parents[I]<0 || Parents[I]>=I) || !Finite(Local[I])) {
            Error=TEXT("Invalid skeleton name, hierarchy, unit scale or transform.");return false;
        }
        Seen.Add(Names[I]);
    }
    Compose(*this,Local,Result);return true;
}
bool FSolver::Initialize(const FSkeleton& InSource,const FSkeleton& InTarget,FString& Error,bool Mannequin) {
    Source=InSource;Target=InTarget;
    UseMannequinGoals=Mannequin;
    IKGuides.Reset();
    if (!Source.World(SR,Error) || !Target.World(TR,Error)) return false;
    SP=Source.Find(TEXT("Hips"));TP=Target.Find(TEXT("pelvis"));
    if (SP<0 || TP<0) { Error=TEXT("Expected RGBMoCap Hips and target pelvis bones.");return false; }
    bool Native=Source.Find(TEXT("LeftShin"))>=0;
    for (int32 Side=0;Side<2;++Side) {
        FString Prefix=Side==0?TEXT("Left"):TEXT("Right"),Suffix=Side==0?TEXT("_l"):TEXT("_r");
        const TCHAR* SourceParts[4]={Native?TEXT("Leg"):TEXT("UpLeg"),Native?TEXT("Shin"):TEXT("Leg"),TEXT("Foot"),TEXT("ToeBase")};
        const TCHAR* TargetParts[4]={TEXT("thigh"),TEXT("calf"),TEXT("foot"),TEXT("ball")};
        for (int32 J=0;J<4;++J) {
            SL[Side][J]=Source.Find(*(Prefix+SourceParts[J]));TL[Side][J]=Target.Find(*(FString(TargetParts[J])+Suffix));
            if (SL[Side][J]<0 || TL[Side][J]<0) { Error=TEXT("Missing supported source or target leg bones.");return false; }
        }
        for (int32 J=0;J<2;++J) {
            Lengths[Side][J]=FVector::Distance(TR[TL[Side][J+1]].GetTranslation(),TR[TL[Side][J]].GetTranslation());
            if (Lengths[Side][J]<1.) { Error=TEXT("Invalid target leg lengths (centimetres expected).");return false; }
        }
    }
    int32 SH=Source.Find(TEXT("Head")),TH=Target.Find(TEXT("head"));
    if (SH<0 || TH<0) { Error=TEXT("Missing head bone for anatomical alignment.");return false; }
    FQuat SB=Basis(SR[SL[1][0]].GetTranslation()-SR[SL[0][0]].GetTranslation(),SR[SH].GetTranslation()-SR[SP].GetTranslation());
    FQuat TB=Basis(TR[TL[1][0]].GetTranslation()-TR[TL[0][0]].GetTranslation(),TR[TH].GetTranslation()-TR[TP].GetTranslation());
    Alignment=TB*SB.Inverse();Alignment.Normalize();
    double SourceSpan=0.,TargetSpan=0.;
    for (int32 Side=0;Side<2;++Side) {
        SourceSpan+=.5*(FVector::Distance(SR[SL[Side][0]].GetTranslation(),SR[SL[Side][1]].GetTranslation())
            +FVector::Distance(SR[SL[Side][1]].GetTranslation(),SR[SL[Side][2]].GetTranslation()));
        TargetSpan+=.5*(Lengths[Side][0]+Lengths[Side][1]);
    }
    SourceSpan+=FVector::Distance(SR[SL[0][0]].GetTranslation(),SR[SL[1][0]].GetTranslation());
    TargetSpan+=FVector::Distance(TR[TL[0][0]].GetTranslation(),TR[TL[1][0]].GetTranslation());
    if (SourceSpan<1. || TargetSpan<1.) { Error=TEXT("Invalid rest leg span.");return false; }
    Gain=TargetSpan/SourceSpan;
    if (UseMannequinGoals) {
        // A shared offset preserves the captured stance width at target scale.
        AnkleOffset=FVector::ZeroVector;
        for (int32 Side=0;Side<2;++Side)
            AnkleOffset+=.5*(TR[TL[Side][2]].GetTranslation()-Gain*Alignment.RotateVector(SR[SL[Side][2]].GetTranslation()));
        const TPair<const TCHAR*,const TCHAR*> Pairs[]={
            {TEXT("ik_foot_l"),TEXT("foot_l")},{TEXT("ik_foot_r"),TEXT("foot_r")},
            {TEXT("ik_hand_gun"),TEXT("hand_r")},{TEXT("ik_hand_l"),TEXT("hand_l")},{TEXT("ik_hand_r"),TEXT("hand_r")}};
        for (const auto& Pair:Pairs) {
            const int32 Guide=Target.Find(Pair.Key),Bone=Target.Find(Pair.Value);
            if (Guide>0 && Bone>0) IKGuides.Emplace(Guide,Bone);
        }
        IKGuides.Sort([](const auto& A,const auto& B) {return A.Key<B.Key;});
    }
    Mapping.Init(INDEX_NONE,TR.Num());AimSource.Init(INDEX_NONE,TR.Num());AimTarget.Init(INDEX_NONE,TR.Num());
    BlendSource.Init(INDEX_NONE,TR.Num());BlendWeight.Init(0.,TR.Num());
    auto Map=[&](FString T,FString S,FString TC=TEXT(""),FString SC=TEXT("")) {
        int32 TI=Target.Find(*T),SI=Source.Find(*S);
        if (TI>=0 && SI>=0) { Mapping[TI]=SI;AimTarget[TI]=Target.Find(*TC);AimSource[TI]=Source.Find(*SC); }
    };
    RotationReference.SetNum(TR.Num());
    for (int32 I=0;I<TR.Num();++I) RotationReference[I]=TR[I].GetRotation();
    Map(TEXT("pelvis"),TEXT("Hips"));
    const TCHAR* Spines52[5]={TEXT("Spine"),TEXT("Spine"),TEXT("Spine1"),TEXT("Spine2"),TEXT("Spine2")};
    const TCHAR* Spines77[5]={TEXT("Spine1"),TEXT("Spine1"),TEXT("Spine2"),TEXT("Chest"),TEXT("Chest")};
    for (int32 I=0;I<5;++I) Map(FString::Printf(TEXT("spine_%02d"),I+1),Native?Spines77[I]:Spines52[I]);
    Map(TEXT("neck_01"),Native?TEXT("Neck1"):TEXT("Neck"));Map(TEXT("neck_02"),Native?TEXT("Neck2"):TEXT("Neck"));Map(TEXT("head"),TEXT("Head"));
    for (int32 Side=0;Side<2;++Side) {
        FString P=Side==0?TEXT("Left"):TEXT("Right"),S=Side==0?TEXT("_l"):TEXT("_r");
        Map(TEXT("clavicle")+S,P+TEXT("Shoulder"));
        const int32 TC=Target.Find(*(TEXT("clavicle")+S)),SC=Source.Find(*(P+TEXT("Shoulder")));
        ClavicleTargets[Side]=TC;
        if (TC>0) {
            ClavicleReferences[Side]=Target.Local[TC].GetRotation();
            const int32 SA=Source.Find(*(P+TEXT("Arm"))),SE=Source.Find(*(P+TEXT("ForeArm")));
            const int32 TA=Target.Find(*(TEXT("upperarm")+S)),TE=Target.Find(*(TEXT("lowerarm")+S));
            if (SC>=0 && SA>=0 && SE>=0 && TA>=0 && TE>=0) {
                const FQuat Lift=ClavicleRestLift(SR[SA].GetTranslation()-SR[SC].GetTranslation(),
                                                SR[SE].GetTranslation()-SR[SA].GetTranslation(),
                                                TR[TA].GetTranslation()-TR[TC].GetTranslation(),
                                                TR[TE].GetTranslation()-TR[TA].GetTranslation());
                const FQuat Parent=TR[Target.Parents[TC]].GetRotation();
                ClavicleReferences[Side]=(Parent.Inverse()*Lift*Parent*ClavicleReferences[Side]).GetNormalized();
            }
        }
        Map(TEXT("upperarm")+S,P+TEXT("Arm"),TEXT("lowerarm")+S,P+TEXT("ForeArm"));
        Map(TEXT("lowerarm")+S,P+TEXT("ForeArm"),TEXT("hand")+S,P+TEXT("Hand"));
        Map(TEXT("hand")+S,P+TEXT("Hand"));
        Map(TEXT("thigh")+S,P+(Native?TEXT("Leg"):TEXT("UpLeg")),TEXT("calf")+S,P+(Native?TEXT("Shin"):TEXT("Leg")));
        Map(TEXT("calf")+S,P+(Native?TEXT("Shin"):TEXT("Leg")),TEXT("foot")+S,P+TEXT("Foot"));
        Map(TEXT("foot")+S,P+TEXT("Foot"));
        Map(TEXT("ball")+S,P+TEXT("ToeBase"));
        const TCHAR* TF[5]={TEXT("thumb"),TEXT("index"),TEXT("middle"),TEXT("ring"),TEXT("pinky")};
        const TCHAR* SF[5]={TEXT("Thumb"),TEXT("Index"),TEXT("Middle"),TEXT("Ring"),TEXT("Pinky")};
        // Calibrate physical palm frames, not the unrelated bone-roll frames.
        // Fingers use the same anatomical reference for all three phalanges,
        // including terminal bones with no child/tip in the source hierarchy.
        int32 SHnd=Source.Find(*(P+TEXT("Hand"))),THnd=Target.Find(*(TEXT("hand")+S));
        auto Palm=[&](const FSkeleton& Sk,const TArray<FTransform>& Rest,int32 H,
                      FName Index,FName Middle,FName Pinky,FQuat& Q) {
            int32 I=Sk.Names.IndexOfByKey(Index),M=Sk.Names.IndexOfByKey(Middle),K=Sk.Names.IndexOfByKey(Pinky);
            if (H<0 || I<0 || M<0 || K<0) return false;
            FVector Forward=Rest[M].GetTranslation()-Rest[H].GetTranslation();
            FVector Across=Rest[I].GetTranslation()-Rest[K].GetTranslation();
            if (!Forward.Normalize() || !Across.Normalize() || FVector::CrossProduct(Forward,Across).SizeSquared()<1.e-6) return false;
            Q=FRotationMatrix::MakeFromXY(Forward,Across).ToQuat();return true;
        };
        FQuat SourcePalm,TargetPalm;
        bool HasPalm=Palm(Source,SR,SHnd,*(P+TEXT("HandIndex1")),*(P+TEXT("HandMiddle1")),*(P+TEXT("HandPinky1")),SourcePalm)
            && Palm(Target,TR,THnd,*(TEXT("index_01")+S),*(TEXT("middle_01")+S),*(TEXT("pinky_01")+S),TargetPalm);
        if (HasPalm) {
            RotationReference[THnd]=(Alignment*SourcePalm*TargetPalm.Inverse()*TR[THnd].GetRotation()).GetNormalized();
            FVector SN=SourcePalm.RotateVector(FVector::UpVector),TN=TargetPalm.RotateVector(FVector::UpVector);
            for (int32 F=0;F<5;++F) for (int32 J=1;J<=3;++J) {
                FString TName=FString::Printf(TEXT("%s_%02d%s"),TF[F],J,*S);
                FString SName=P+TEXT("Hand")+SF[F]+FString::FromInt(J);
                int32 TI=Target.Find(*TName),SI=Source.Find(*SName);
                if (TI<0 || SI<0) continue;
                // RGB MoCap's UE stream maps Blender longitudinal +Y to +Z.
                // MetaHuman's phalanges use mirrored +/-X. Determine the sign
                // from reference joint geometry; never use corrective children.
                int32 Next=J<3?Target.Find(*FString::Printf(TEXT("%s_%02d%s"),TF[F],J+1,*S)):INDEX_NONE;
                int32 Prev=Target.Parents[TI];
                if (Next<0 && Prev<0) continue;
                FVector Geometric=Next>=0?TR[Next].GetTranslation()-TR[TI].GetTranslation():TR[TI].GetTranslation()-TR[Prev].GetTranslation();
                FVector TFwd=TR[TI].GetRotation().RotateVector(FVector::ForwardVector);
                if (FVector::DotProduct(TFwd,Geometric)<0.) TFwd=-TFwd;
                FVector SFwd=SR[SI].GetRotation().RotateVector(FVector::UpVector);
                if (Geometric.IsNearlyZero() || FVector::CrossProduct(SFwd,SN).SizeSquared()<1.e-6 || FVector::CrossProduct(TFwd,TN).SizeSquared()<1.e-6) continue;
                FQuat SBone=FRotationMatrix::MakeFromXZ(SFwd,SN).ToQuat();
                FQuat TBone=FRotationMatrix::MakeFromXZ(TFwd,TN).ToQuat();
                Mapping[TI]=SI;
                RotationReference[TI]=(Alignment*SBone*TBone.Inverse()*TR[TI].GetRotation()).GetNormalized();
            }
        }
        int32 SA=SL[Side][2],TA=TL[Side][2],TT=TL[Side][3];
        FVector Heel=TR[TA].GetTranslation(),Toe=TR[TT].GetTranslation();
        FVector Forward=Toe-Heel;Forward.Z=0.;
        Heel-=.25*Forward;Heel.Z=Toe.Z=TR[0].GetTranslation().Z;
        Markers[Side][0]=TR[TA].InverseTransformPosition(Heel);
        Toe+=.2*Forward;
        Markers[Side][1]=TR[TT].InverseTransformPosition(Toe);
    }
    // Interpolate global rest-to-pose deltas across unequal torso chain counts.
    auto Chain=[&](const TArray<int32>& Src,const TArray<int32>& Dst) {
        TArray<double> SD,TD;SD.Add(0.);TD.Add(0.);
        for (int32 I=1;I<Src.Num();++I) SD.Add(SD.Last()+FVector::Distance(SR[Src[I-1]].GetTranslation(),SR[Src[I]].GetTranslation()));
        for (int32 I=1;I<Dst.Num();++I) TD.Add(TD.Last()+FVector::Distance(TR[Dst[I-1]].GetTranslation(),TR[Dst[I]].GetTranslation()));
        if (SD.Last()<1.e-4 || TD.Last()<1.e-4) return;
        for (int32 I=1;I<Dst.Num();++I) {
            double U=TD[I]/TD.Last()*SD.Last();int32 J=1;
            while (J<SD.Num()-1 && SD[J]<U) ++J;
            Mapping[Dst[I]]=Src[J-1];BlendSource[Dst[I]]=Src[J];
            BlendWeight[Dst[I]]=FMath::Clamp((U-SD[J-1])/FMath::Max(SD[J]-SD[J-1],1.e-6),0.,1.);
        }
    };
    TArray<int32> SrcTorso{SP},DstTorso{TP};
    for (const TCHAR* N:Native?TArray<const TCHAR*>{TEXT("Spine1"),TEXT("Spine2"),TEXT("Chest")}:TArray<const TCHAR*>{TEXT("Spine"),TEXT("Spine1"),TEXT("Spine2")}) {
        int32 I=Source.Find(N);if (I>=0) SrcTorso.Add(I);
    }
    for (int32 I=1;I<=5;++I) {int32 J=Target.Find(*FString::Printf(TEXT("spine_%02d"),I));if (J>=0) DstTorso.Add(J);}
    Chain(SrcTorso,DstTorso);
    TArray<int32> SrcNeck{SrcTorso.Last()},DstNeck{DstTorso.Last()};
    for (const TCHAR* N:Native?TArray<const TCHAR*>{TEXT("Neck1"),TEXT("Neck2"),TEXT("Head")}:TArray<const TCHAR*>{TEXT("Neck"),TEXT("Head")}) {
        int32 I=Source.Find(N);if (I>=0) SrcNeck.Add(I);
    }
    for (const TCHAR* N:{TEXT("neck_01"),TEXT("neck_02"),TEXT("head")}) {int32 I=Target.Find(N);if (I>=0) DstNeck.Add(I);}
    Chain(SrcNeck,DstNeck);
    return true;
}
bool FSolver::Evaluate(const TArray<FTransform>& SourceLocal,const FSettings& Settings,TArray<FTransform>& Output,FDiagnostics& D,FString& Error) const {
    FSkeleton Current=Source;Current.Local=SourceLocal;TArray<FTransform> SW;
    if (!Current.World(SW,Error)) return false;
    Output=Target.Local;TArray<FTransform> W;Compose(Target,Output,W);
    FVector Pelvis=TR[TP].GetTranslation()+Gain*Alignment.RotateVector(SW[SP].GetTranslation()-SR[SP].GetTranslation());
    // Component-space locomotion lives in pelvis; keep the separate root fixed.
    Compose(Target,Output,W);
    for (int32 I=1;I<W.Num();++I) {
        int32 P=Target.Parents[I],SI=Mapping[I];
        const int32 ClavicleSide=I==ClavicleTargets[0]?0:I==ClavicleTargets[1]?1:INDEX_NONE;
        if (SI>=0 && Source.Parents[SI]>=0 && ClavicleSide!=INDEX_NONE) {
            // Express the captured local change in the target chest axes,
            // then apply it to the target clavicle's own reference rotation.
            const FQuat Bridge=TR[P].GetRotation().Inverse()*Alignment*SR[Source.Parents[SI]].GetRotation();
            const FQuat Delta=SourceLocal[SI].GetRotation()*Source.Local[SI].GetRotation().Inverse();
            Output[I].SetRotation((Bridge*Delta*Bridge.Inverse()*ClavicleReferences[ClavicleSide]).GetNormalized());
        } else if (SI>=0) {
            FQuat Delta=Alignment*SW[SI].GetRotation()*SR[SI].GetRotation().Inverse()*Alignment.Inverse();
            if (BlendSource[I]>=0) {
                int32 Other=BlendSource[I];FQuat D2=Alignment*SW[Other].GetRotation()*SR[Other].GetRotation().Inverse()*Alignment.Inverse();
                Delta=FQuat::Slerp(Delta,D2,BlendWeight[I]).GetNormalized();
            }
            FQuat Rotation=Delta*RotationReference[I];
            int32 SC=AimSource[I],TC=AimTarget[I];
            if (SC>=0 && TC>=0) {
                FVector Before=Delta.RotateVector(TR[TC].GetTranslation()-TR[I].GetTranslation());
                FVector After=Alignment.RotateVector(SW[SC].GetTranslation()-SW[SI].GetTranslation());
                if (!Before.IsNearlyZero() && !After.IsNearlyZero()) Rotation=FQuat::FindBetweenNormals(Before.GetSafeNormal(),After.GetSafeNormal())*Rotation;
            }
            Output[I].SetRotation((W[P].GetRotation().Inverse()*Rotation).GetNormalized());
        }
        W[I]=Output[I]*W[P];
        if (I==TP) { Output[I].SetTranslation(W[P].InverseTransformPosition(Pelvis));W[I]=Output[I]*W[P]; }
    }
    FVector Goals[2];FQuat FootRotations[2];
    const double Floor=TR[0].GetTranslation().Z+Settings.FloorOffset;
    double Lower=-.3*(Lengths[0][0]+Lengths[0][1]),Upper=-Lower;
    for (int32 Side=0;Side<2;++Side) {
        int32 SA=SL[Side][2],TA=TL[Side][2];
        Goals[Side]=TR[TA].GetTranslation()+Gain*Alignment.RotateVector(SW[SA].GetTranslation()-SR[SA].GetTranslation());
        if (UseMannequinGoals) {
            const FVector Planar=AnkleOffset+Gain*Alignment.RotateVector(SW[SA].GetTranslation());
            Goals[Side].X=Planar.X;Goals[Side].Y=Planar.Y;
        }
        FootRotations[Side]=W[TA].GetRotation();
        if (Settings.Ground) {
            double Low=DBL_MAX;
            Low=Goals[Side].Z+FootRotations[Side].RotateVector(Markers[Side][0]).Z;
            int32 Ball=TL[Side][3];FTransform Ankle(FootRotations[Side],Goals[Side]);
            FTransform Toe=Output[Ball]*Ankle;
            Low=FMath::Min(Low,Toe.TransformPosition(Markers[Side][1]).Z);
            Goals[Side].Z+=FMath::Max(0.,Floor-Low);
        }
        FVector Hip=W[TL[Side][0]].GetTranslation(),Diff=Hip-Goals[Side];
        double Max=Lengths[Side][0]+Lengths[Side][1]-.01;
        if (Diff.SizeSquared2D()<Max*Max) {
            double Height=FMath::Sqrt(Max*Max-Diff.SizeSquared2D());
            Lower=FMath::Max(Lower,-Diff.Z-Height);Upper=FMath::Min(Upper,-Diff.Z+Height);
        }
    }
    double Shift=Lower<=Upper?FMath::Clamp(0.,Lower,Upper):FMath::Clamp(.5*(Lower+Upper),-.3*(Lengths[0][0]+Lengths[0][1]),.3*(Lengths[0][0]+Lengths[0][1]));
    if (FMath::Abs(Shift)>1.e-7) { Pelvis.Z+=Shift;SetWorldPosition(Target,TP,Pelvis,Output,W); }
    D.PelvisAdjustment=Shift;D.TravelScale=Gain;D.MinimumSole=DBL_MAX;D.ReachError=0.;
    for (int32 Side=0;Side<2;++Side) {
        int32 H=TL[Side][0],K=TL[Side][1],A=TL[Side][2];
        FVector Hip=W[H].GetTranslation(),Direction=Goals[Side]-Hip;
        double L1=Lengths[Side][0],L2=Lengths[Side][1],Distance=FMath::Clamp(Direction.Size(),FMath::Abs(L1-L2)+.01,L1+L2-.01);
        FVector Axis=Direction.GetSafeNormal(UE_SMALL_NUMBER,FVector(0,0,-1));
        FVector SrcLeg=SW[SL[Side][2]].GetTranslation()-SW[SL[Side][0]].GetTranslation();
        FVector SrcUpper=SW[SL[Side][1]].GetTranslation()-SW[SL[Side][0]].GetTranslation();
        FVector Bend=Alignment.RotateVector(SrcUpper-SrcLeg.GetSafeNormal()*FVector::DotProduct(SrcUpper,SrcLeg.GetSafeNormal()));
        FVector SourceAxis=Alignment.RotateVector(SrcLeg.GetSafeNormal());
        Bend=FQuat::FindBetweenNormals(SourceAxis,Axis).RotateVector(Bend);
        Bend-=Axis*FVector::DotProduct(Bend,Axis);
        FVector Forward=Alignment.RotateVector(Basis(SR[SL[1][0]].GetTranslation()-SR[SL[0][0]].GetTranslation(),SR[Source.Find(TEXT("Head"))].GetTranslation()-SR[SP].GetTranslation()).RotateVector(FVector::ForwardVector));
        FQuat PelvisDelta=Alignment*SW[SP].GetRotation()*SR[SP].GetRotation().Inverse()*Alignment.Inverse();
        Forward=PelvisDelta.RotateVector(Forward);Forward-=Axis*FVector::DotProduct(Forward,Axis);
        if (Forward.IsNearlyZero()) Forward=FVector::CrossProduct(Axis,FVector::RightVector);
        double Confidence=FMath::Clamp(Bend.Size()/(.025*(L1+L2)),0.,1.);
        Confidence=Confidence*Confidence*(3.-2.*Confidence);
        Bend=(Confidence*Bend.GetSafeNormal()+(1.-Confidence)*Forward.GetSafeNormal()).GetSafeNormal(UE_SMALL_NUMBER,FVector(1,0,0));
        double Along=(L1*L1-L2*L2+Distance*Distance)/(2.*Distance);
        FVector Knee=Hip+Along*Axis+FMath::Sqrt(FMath::Max(0.,L1*L1-Along*Along))*Bend;
        FVector OldUpper=W[K].GetTranslation()-Hip;
        SetWorldRotation(Target,H,FQuat::FindBetweenNormals(OldUpper.GetSafeNormal(),(Knee-Hip).GetSafeNormal())*W[H].GetRotation(),Output,W);
        FVector Ankle=Hip+Distance*Axis,OldLower=W[A].GetTranslation()-W[K].GetTranslation();
        SetWorldRotation(Target,K,FQuat::FindBetweenNormals(OldLower.GetSafeNormal(),(Ankle-W[K].GetTranslation()).GetSafeNormal())*W[K].GetRotation(),Output,W);
        SetWorldRotation(Target,A,FootRotations[Side],Output,W);
        D.ReachError=FMath::Max(D.ReachError,FVector::Distance(W[A].GetTranslation(),Goals[Side]));
        D.MinimumSole=FMath::Min(D.MinimumSole,W[A].TransformPosition(Markers[Side][0]).Z-Floor);
        D.MinimumSole=FMath::Min(D.MinimumSole,W[TL[Side][3]].TransformPosition(Markers[Side][1]).Z-Floor);
    }
    for (const auto& Guide:IKGuides) {
        Output[Guide.Key]=W[Guide.Value].GetRelativeTransform(W[Target.Parents[Guide.Key]]);
        Compose(Target,Output,W);
    }
    for (const auto& T:Output) if (!Finite(T)) { Error=TEXT("Retarget produced an invalid transform.");return false; }
    return true;
}
}
