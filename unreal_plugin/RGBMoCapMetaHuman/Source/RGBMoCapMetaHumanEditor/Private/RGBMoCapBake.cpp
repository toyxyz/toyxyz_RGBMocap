#line 1 "Source/RGBMoCapMetaHumanEditor/Private/RGBMoCapBake.cpp"
#include "RGBMoCapBake.h"

#include "Animation/AnimData/IAnimationDataController.h"
#include "Animation/AnimData/CurveIdentifier.h"
#include "Animation/AnimSequence.h"
#include "Animation/Skeleton.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Async/Async.h"
#include "Curves/RichCurve.h"
#include "Engine/SkeletalMesh.h"
#include "GenericPlatform/GenericPlatformHttp.h"
#include "HAL/FileManager.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "UObject/SavePackage.h"

namespace RGBMoCap {
namespace {
constexpr int32 MaxBytes=128*1024*1024;
constexpr int32 MaxFrames=54000;
constexpr int64 MaxTargetKeys=12000000;
constexpr int32 FaceChannels=52;
constexpr int32 HeaderBytes=20;
const FName HeadControlSwitch(TEXT("HeadControlSwitch"));

class FBakeResponseArchive : public FArchive {
public:
    TArray<uint8> Bytes;

    void Serialize(void* Data,int64 Count) override {
        if (Count<0 || Count>MaxBytes-Bytes.Num()) {SetError();return;}
        Bytes.Append(static_cast<const uint8*>(Data),static_cast<int32>(Count));
    }
    FString GetArchiveName() const override {return TEXT("RGB MoCap bake response");}
};

bool ParseHeader(const TArray<uint8>& Data,int32 SourceBones,FBakedTracks& Out,bool& HasFace,FString& Error) {
    if (Data.Num()<HeaderBytes || Data.Num()>MaxBytes) {
        Error=TEXT("Invalid bake payload.");return false;
    }
    HasFace=FMemory::Memcmp(Data.GetData(),"RMB2",4)==0;
    if (!HasFace && FMemory::Memcmp(Data.GetData(),"RMB1",4)!=0) {
        Error=TEXT("Unknown bake payload version.");return false;
    }
    uint32 Header[4];
    FMemory::Memcpy(Header,Data.GetData()+4,sizeof(Header));
    const uint32 Num=Header[0],Den=Header[1],Frames=Header[2],Bones=Header[3];
    if (!Num || !Den || Num>24000000 || Den>100000 || Num<Den || Num>240*Den ||
        Frames<2 || Frames>MaxFrames || Bones!=static_cast<uint32>(SourceBones) ||
        Bones<1 || Bones>256 ||
        Data.Num()!=HeaderBytes+int64(Frames)*(Bones*7+(HasFace?FaceChannels:0))*sizeof(float)) {
        Error=TEXT("The take is incomplete or its frame rate or skeleton changed.");return false;
    }
    Out.Rate=FFrameRate(Num,Den);
    Out.Frames=Frames;
    return true;
}
}

bool SolveBake(const FRGBMoCapBakeContext& Context,const TArray<uint8>& Payload,FBakedTracks& Out,FString& Error) {
    const int32 SourceBones=Context.Source.Names.Num();
    bool HasFace=false;
    if (!ParseHeader(Payload,SourceBones,Out,HasFace,Error)) return false;
    const int32 FaceCurves=Context.FaceCurveNames.Num();
    const int64 FaceKeys=int64(Out.Frames)*FaceCurves;
    Out.FaceWarning=Context.FaceWarning;
    if (HasFace && (!Context.FaceRequested || FaceCurves==0 || Context.FaceMatrix.Num()!=int64(FaceCurves)*FaceChannels)) {
        Error=TEXT("Unexpected face samples for this target.");return false;
    }
    if (HasFace && FaceKeys<=MaxTargetKeys) Out.FaceKeys.SetNumZeroed(FaceKeys);
    if (HasFace && Out.FaceKeys.IsEmpty()) Out.FaceWarning=TEXT("Too many face keys for this take.");
    if (!HasFace && Context.FaceRequested && Out.FaceWarning.IsEmpty())
        Out.FaceWarning=TEXT("No saved face samples for this take.");
    Out.Bones=Context.RawBones;
    const int64 Total=int64(Out.Frames)*Out.Bones;
    if (Out.Bones<1 || Out.Bones>Context.Target.Names.Num() || Total>MaxTargetKeys) {
        Error=TEXT("This take is too large for the target skeleton.");return false;
    }
    FSolver Solver;
    if (!Solver.Initialize(Context.Source,Context.Target,Error,Context.Mannequin)) return false;
    Out.Keys.SetNumUninitialized(Total);
    TArray<FTransform> Input,Pose;
    Input.SetNum(SourceBones);
    const uint8* Reader=Payload.GetData()+HeaderBytes;
    for (int32 Frame=0;Frame<Out.Frames;++Frame) {
        for (FTransform& Bone:Input) {
            float V[7];
            FMemory::Memcpy(V,Reader,sizeof(V));
            Reader+=sizeof(V);
            for (float Value:V) if (!FMath::IsFinite(Value)) {
                Error=TEXT("The take contains invalid transforms.");return false;
            }
            const FQuat Q(V[3],V[4],V[5],V[6]);
            if (!FMath::IsNearlyEqual(Q.SizeSquared(),1.,.01)) {
                Error=TEXT("The take contains an invalid rotation.");return false;
            }
            Bone=FTransform(Q.GetNormalized(),FVector(V[0],V[1],V[2]));
        }
        FDiagnostics Diagnostics;
        if (!Solver.Evaluate(Input,Context.Settings,Pose,Diagnostics,Error)) return false;
        for (int32 Bone=0;Bone<Out.Bones;++Bone) Out.Keys[Frame*Out.Bones+Bone]=FTransform3f(Pose[Bone]);
        if (HasFace) {
            float Coefficients[FaceChannels];
            FMemory::Memcpy(Coefficients,Reader,sizeof(Coefficients));
            Reader+=sizeof(Coefficients);
            if (!Out.FaceKeys.IsEmpty()) {
                for (float Value:Coefficients) if (!FMath::IsFinite(Value) || Value<0.f || Value>1.f) {
                    Out.FaceKeys.Reset();Out.FaceWarning=TEXT("The take contains invalid face coefficients.");break;
                }
            }
            if (!Out.FaceKeys.IsEmpty()) {
                auto Mapped=MakeArrayView(Out.FaceKeys.GetData()+Frame*FaceCurves,FaceCurves);
                URGBMoCapConnectComponent::MapFace(MakeArrayView(Coefficients,FaceChannels),Context.FaceMatrix,Mapped);
                for (float Value:Mapped) if (!FMath::IsFinite(Value)) {
                    Out.FaceKeys.Reset();Out.FaceWarning=TEXT("The target face mapping contains invalid curves.");break;
                }
            }
        }
    }
    return true;
}

FString FaceBakePath(const FString& BodyPath) {
    return FPackageName::ObjectPathToPackageName(BodyPath)+TEXT("_Face.")+
        FPackageName::ObjectPathToObjectName(BodyPath)+TEXT("_Face");
}

bool CheckBakeDestination(const FString& ObjectPath,FString& Filename,FString& Error) {
    const FString PackageName=FPackageName::ObjectPathToPackageName(ObjectPath);
    if (!FPackageName::IsValidObjectPath(ObjectPath) || !PackageName.StartsWith(TEXT("/Game/")) ||
        !FPackageName::TryConvertLongPackageNameToFilename(PackageName,Filename,FPackageName::GetAssetPackageExtension()) ||
        FPackageName::DoesPackageExist(PackageName) || FindObject<UObject>(nullptr,*ObjectPath)) {
        Error=TEXT("Choose an unused Animation Sequence name under /Game. Already used or invalid: ")+ObjectPath;
        return false;
    }
    return true;
}

UAnimSequence* SaveBake(const FRGBMoCapBakeContext& Context,const FBakedTracks& Animation,
                        const FString& ObjectPath,FString& Error,bool Face) {
    check(IsInGameThread());
    auto* Mesh=(Face?Context.FaceMesh:Context.Mesh).Get();
    auto* Skeleton=Mesh?Mesh->GetSkeleton():nullptr;
    const int32 BoneCount=Face?0:Animation.Bones;
    const bool HasKeys=Face?
        !Context.FaceCurveNames.IsEmpty() && Animation.FaceKeys.Num()==int64(Animation.Frames)*Context.FaceCurveNames.Num():
        BoneCount==Context.RawBones && Animation.Keys.Num()==int64(Animation.Frames)*BoneCount;
    if (!Skeleton || Animation.Frames<2 || !HasKeys) {
        Error=TEXT("The target skeleton or bake data is unavailable.");return nullptr;
    }
    FString Filename;
    if (!CheckBakeDestination(ObjectPath,Filename,Error)) return nullptr;
    const auto& Ref=Skeleton->GetReferenceSkeleton();
    for (int32 I=0;I<BoneCount;++I) {
        const FName Name=Context.Target.Names[I];
        const int32 J=Ref.FindRawBoneIndex(Name);
        const int32 P=Context.Target.Parents[I];
        if (J==INDEX_NONE || (P==INDEX_NONE ? Ref.GetRawParentIndex(J)!=INDEX_NONE :
            Ref.GetRawParentIndex(J)!=Ref.FindRawBoneIndex(Context.Target.Names[P]))) {
            Error=TEXT("The target mesh hierarchy does not match its Skeleton asset.");return nullptr;
        }
    }
    const FString PackageName=FPackageName::ObjectPathToPackageName(ObjectPath);
    const FString AssetName=FPackageName::ObjectPathToObjectName(ObjectPath);
    auto* Package=CreatePackage(*PackageName);
    if (FindObjectFast<UObject>(Package,*AssetName)) {
        Error=TEXT("An asset already exists at this location.");return nullptr;
    }
    auto* Sequence=NewObject<UAnimSequence>(Package,*AssetName,RF_Public|RF_Standalone);
    auto Discard=[&](const TCHAR* Message) -> UAnimSequence* {
        Sequence->ClearFlags(RF_Public|RF_Standalone);
        Sequence->Rename(nullptr,GetTransientPackage(),REN_DontCreateRedirectors|REN_NonTransactional);
        Sequence->MarkAsGarbage();
        Package->SetDirtyFlag(false);
        Error=Message;
        return nullptr;
    };
    Sequence->SetSkeleton(Skeleton);
    Sequence->SetPreviewMesh(Mesh,false);
    auto& Controller=Sequence->GetController();
    Controller.InitializeModel();
    Controller.OpenBracket(NSLOCTEXT("RGBMoCap","BakeAnimation","Bake RGB MoCap Animation"),false);
    Controller.SetFrameRate(Animation.Rate,false);
    Controller.SetNumberOfFrames(Animation.Frames-1,false);
    if (Face) {
        TArray<FRichCurveKey> CurveKeys;
        CurveKeys.SetNum(Animation.Frames);
        const auto WriteCurve=[&](FName Name,int32 Index) {
            const FAnimationCurveIdentifier Curve(Name,ERawCurveTrackTypes::RCT_Float);
            for (int32 Frame=0;Frame<Animation.Frames;++Frame) {
                auto& Key=CurveKeys[Frame];
                Key.Time=static_cast<float>(Frame*Animation.Rate.AsInterval());
                Key.Value=Index==INDEX_NONE?0.f:Animation.FaceKeys[Frame*Context.FaceCurveNames.Num()+Index];
                Key.InterpMode=RCIM_Linear;
            }
            return Controller.AddCurve(Curve,AACF_DefaultCurve,false) && Controller.SetCurveKeys(Curve,CurveKeys,false);
        };
        for (int32 Curve=0;Curve<Context.FaceCurveNames.Num();++Curve) {
            const FName Name=Context.FaceCurveNames[Curve];
            if (Name!=HeadControlSwitch && !WriteCurve(Name,Curve)) {
                Controller.CloseBracket(false);
                return Discard(TEXT("Unreal could not write a MetaHuman face curve."));
            }
        }
        if (!WriteCurve(HeadControlSwitch,INDEX_NONE)) {
            Controller.CloseBracket(false);
            return Discard(TEXT("Unreal could not write the MetaHuman head control curve."));
        }
    } else {
        TArray<FVector3f> Positions,Scales;
        TArray<FQuat4f> Rotations;
        Positions.SetNumUninitialized(Animation.Frames);
        Scales.SetNumUninitialized(Animation.Frames);
        Rotations.SetNumUninitialized(Animation.Frames);
        for (int32 Bone=0;Bone<BoneCount;++Bone) {
            const FName Name=Context.Target.Names[Bone];
            for (int32 Frame=0;Frame<Animation.Frames;++Frame) {
                const auto& T=Animation.Keys[Frame*Animation.Bones+Bone];
                Positions[Frame]=T.GetTranslation();
                Scales[Frame]=T.GetScale3D();
                Rotations[Frame]=T.GetRotation();
                if (Frame>0 && (Rotations[Frame-1]|Rotations[Frame])<0.f) Rotations[Frame]*=-1.f;
            }
            if (!Controller.AddBoneCurve(Name,false) || !Controller.SetBoneTrackKeys(Name,Positions,Rotations,Scales,false)) {
                Controller.CloseBracket(false);
                return Discard(TEXT("Unreal could not write an animation bone track."));
            }
        }
    }
    Controller.NotifyPopulated();
    Controller.CloseBracket(false);
    Sequence->MarkPackageDirty();
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(Filename),true);
    FSavePackageArgs Args;
    Args.TopLevelFlags=RF_Public|RF_Standalone;
    if (!UPackage::SavePackage(Package,Sequence,*Filename,Args)) {
        return Discard(TEXT("Unreal could not save the Animation Sequence."));
    }
    FAssetRegistryModule::AssetCreated(Sequence);
    return Sequence;
}

void RequestBake(FRGBMoCapBakeContext Context,TFunction<void(FBakedTracks,FString)> Complete) {
    const auto URL=FString::Printf(TEXT("http://127.0.0.1:%d/take?subject=%s&rig=%s&take=%s"),
        Context.Port,*FGenericPlatformHttp::UrlEncode(Context.Subject.ToString()),
        *FGenericPlatformHttp::UrlEncode(Context.Rig),*FGenericPlatformHttp::UrlEncode(Context.Take));
    auto Request=FHttpModule::Get().CreateRequest();
    auto Buffer=MakeShared<FBakeResponseArchive>();
    Request->SetVerb(TEXT("GET"));
    Request->SetURL(URL);
    Request->SetHeader(TEXT("Authorization"),TEXT("Bearer ")+Context.Token);
    if (!Context.FaceCurveNames.IsEmpty()) Request->SetHeader(TEXT("X-RGBMoCap-Face"),TEXT("1"));
    Request->SetTimeout(300.f);
    if (!Request->SetResponseBodyReceiveStream(Buffer)) {
        Complete({},TEXT("The HTTP transport could not start the bake transfer."));
        return;
    }
    auto Callback=MakeShared<TFunction<void(FBakedTracks,FString)>>(MoveTemp(Complete));
    Request->OnProcessRequestComplete().BindLambda(
        [Context=MoveTemp(Context),Buffer,Callback](FHttpRequestPtr,FHttpResponsePtr Response,bool Success) mutable {
            const int32 Code=Response?Response->GetResponseCode():0;
            if (!Success || Code!=200) {
                (*Callback)({},Code==409?TEXT("The connected take changed. Select Bake again."):
                    TEXT("Could not download the take. Keep RGB MoCap and the selected take connected."));
                return;
            }
            Async(EAsyncExecution::ThreadPool,[Context=MoveTemp(Context),Buffer,Callback]() mutable {
                FBakedTracks Tracks;
                FString Error;
                if (!SolveBake(Context,Buffer->Bytes,Tracks,Error)) {
                    Tracks={};
                    if (Error.IsEmpty()) Error=TEXT("The target pose could not be solved.");
                }
                Buffer->Bytes.Reset();
                AsyncTask(ENamedThreads::GameThread,[Callback,Tracks=MoveTemp(Tracks),Error=MoveTemp(Error)]() mutable {
                    (*Callback)(MoveTemp(Tracks),MoveTemp(Error));
                });
            });
        });
    if (!Request->ProcessRequest()) {
        Request->OnProcessRequestComplete().Unbind();
        (*Callback)({},TEXT("Unreal could not start the local bake request."));
    }
}
}
