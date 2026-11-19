// RGB MoCap authored sender. Uses the installed engine's official provider.
// No receiver plugin, inference, editor asset creation, or mesh transfer.
#include "Modules/ModuleManager.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformTime.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "LiveLinkProvider.h"
#include "Roles/LiveLinkAnimationRole.h"
#include "Roles/LiveLinkAnimationTypes.h"

class FRGBMoCapLiveLink : public IModuleInterface {
    struct FActor {
        FString Rig, Label;
        TArray<FName> Names;
        TArray<int32> Parents;
        TArray<FTransform> Rest, Pose;
        TArray<float> Lengths;
        int32 Frame = 0;
    };
    TSharedPtr<ILiveLinkProvider> Provider;
    TMap<FName, FActor> Actors;
    FSocket* Socket = nullptr;
    TArray<uint8> Incoming, Outgoing;
    int32 OutOffset = 0;
    FTSTicker::FDelegateHandle Ticker;
    double LastInput = 0, LastPose = 0, LastReference = 0, LastStatus = 0;
    bool Quitting = false;
    static constexpr int32 Limit = 4 * 1024 * 1024;

    void Fail(const TCHAR* Message) {
        UE_LOG(LogTemp, Error, TEXT("RGB MoCap Live Link: %s"), Message);
        Quitting = true;
        FPlatformMisc::RequestExitWithStatus(false, 1);
    }
    static bool Number(const TSharedPtr<FJsonValue>& Value, double& Result) {
        return Value.IsValid() && Value->TryGetNumber(Result) && FMath::IsFinite(Result);
    }
    static bool Transforms(const TArray<TSharedPtr<FJsonValue>>& Values, TArray<FTransform>& Result, int32 Count) {
        if (Values.Num() != Count) return false;
        for (const auto& Value : Values) {
            const TArray<TSharedPtr<FJsonValue>>* V = nullptr;
            if (!Value->TryGetArray(V) || V->Num() != 7) return false;
            double N[7];
            for (int32 I = 0; I < 7; ++I) if (!Number((*V)[I], N[I])) return false;
            FQuat Q(N[3], N[4], N[5], N[6]);
            if (!FMath::IsNearlyEqual(Q.SizeSquared(), 1., .01)) return false;
            Q.Normalize();
            Result.Emplace(Q, FVector(N[0], N[1], N[2]), FVector::OneVector);
        }
        return true;
    }
    bool Parse(const FString& Text) {
        TSharedPtr<FJsonObject> Root;
        if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid()) return false;
        double Version = 0;
        const TArray<TSharedPtr<FJsonValue>>* People = nullptr;
        if (!Root->TryGetNumberField(TEXT("version"), Version) || Version != 1
            || !Root->TryGetArrayField(TEXT("actors"), People) || People->Num() > 128) return false;
        TMap<FName, FActor> Next;
        for (const auto& Item : *People) {
            const TSharedPtr<FJsonObject>* O = nullptr;
            if (!Item->TryGetObject(O) || !O->IsValid()) return false;
            FString Subject, Rig, Label;
            double Frame = 0;
            const TArray<TSharedPtr<FJsonValue>> *Names=nullptr, *Parents=nullptr, *Rest=nullptr, *Lengths=nullptr, *Pose=nullptr;
            if (!(*O)->TryGetStringField(TEXT("subject"), Subject) || !Subject.StartsWith(TEXT("RGBMoCap_")) || Subject.Len() > 90
                || !(*O)->TryGetStringField(TEXT("rig"), Rig) || Rig.Len() != 64
                || !(*O)->TryGetStringField(TEXT("label"), Label) || Label.Len() > 120
                || !(*O)->TryGetNumberField(TEXT("frame"), Frame) || !FMath::IsFinite(Frame) || Frame < 0 || Frame > MAX_int32
                || !(*O)->TryGetArrayField(TEXT("names"), Names) || Names->Num() < 1 || Names->Num() > 256
                || !(*O)->TryGetArrayField(TEXT("parents"), Parents) || Parents->Num() != Names->Num()
                || !(*O)->TryGetArrayField(TEXT("rest"), Rest)
                || !(*O)->TryGetArrayField(TEXT("lengths"), Lengths) || Lengths->Num() != Names->Num()
                || !(*O)->TryGetArrayField(TEXT("pose"), Pose)) return false;
            FName Key(*Subject);
            if (Next.Contains(Key)) return false;
            FActor A;
            A.Rig = Rig; A.Label = Label; A.Frame = (int32)Frame;
            for (int32 I = 0; I < Names->Num(); ++I) {
                FString Name; double Parent=0, Length=0;
                if (!(*Names)[I]->TryGetString(Name) || Name.IsEmpty() || Name.Len() > 120
                    || !Number((*Parents)[I], Parent) || Parent != FMath::FloorToDouble(Parent)
                    || (I == 0 ? Parent != -1 : Parent < 0 || Parent >= I)
                    || !Number((*Lengths)[I], Length) || Length <= 0) return false;
                FName Bone(*Name);
                if (A.Names.Contains(Bone)) return false;
                A.Names.Add(Bone); A.Parents.Add((int32)Parent); A.Lengths.Add((float)Length);
            }
            if (!Transforms(*Rest, A.Rest, Names->Num()) || !Transforms(*Pose, A.Pose, Names->Num())) return false;
            Next.Add(Key, MoveTemp(A));
        }
        // Commit only after the entire snapshot has been checked.
        for (const auto& Pair : Actors) if (!Next.Contains(Pair.Key)) Remove(Pair.Key);
        for (const auto& Pair : Next) {
            const auto* Old = Actors.Find(Pair.Key);
            if (!Old || Old->Rig != Pair.Value.Rig) {
                if (Old) Remove(Pair.Key);
                Declare(Pair.Key, Pair.Value, false);
                Declare(FName(*(Pair.Key.ToString()+TEXT("_Reference"))), Pair.Value, true);
            }
        }
        Actors = MoveTemp(Next);
        return true;
    }
    void Remove(FName Name) {
        Provider->RemoveSubject(Name);
        Provider->RemoveSubject(FName(*(Name.ToString()+TEXT("_Reference"))));
    }
    void Declare(FName Name, const FActor& A, bool Reference) {
        FLiveLinkStaticDataStruct Data(FLiveLinkSkeletonStaticData::StaticStruct());
        auto* S = Data.Cast<FLiveLinkSkeletonStaticData>();
        S->BoneNames = A.Names; S->BoneParents = A.Parents;
        S->PropertyNames.Add(TEXT("RGBMocapFrame"));
        if (Reference) for (const auto& Bone : A.Names)
            S->PropertyNames.Add(FName(*(TEXT("BoneLength_")+Bone.ToString())));
        Provider->UpdateSubjectStaticData(Name, ULiveLinkAnimationRole::StaticClass(), MoveTemp(Data));
    }
    void Send(FName Name, const FActor& A, bool Reference) {
        FLiveLinkFrameDataStruct Data(FLiveLinkAnimationFrameData::StaticStruct());
        auto* D = Data.Cast<FLiveLinkAnimationFrameData>();
        D->Transforms = Reference ? A.Rest : A.Pose;
        D->PropertyValues.Add(Reference ? -1.f : (float)A.Frame);
        if (Reference) for (float Length : A.Lengths) D->PropertyValues.Add(Length);
        D->MetaData.StringMetaData.Add(TEXT("SourceFrame"), Reference ? TEXT("-1") : FString::FromInt(A.Frame));
        D->MetaData.StringMetaData.Add(TEXT("RigId"), A.Rig);
        D->MetaData.StringMetaData.Add(TEXT("Person"), A.Label);
        D->MetaData.StringMetaData.Add(TEXT("Units"), TEXT("centimetres"));
        Provider->UpdateSubjectFrameData(Name, MoveTemp(Data));
    }
    bool Tick(float) {
        if (Quitting) return true;
        double Now = FPlatformTime::Seconds();
        uint32 Pending = 0;
        int32 ReadBudget = Limit * 2;
        while (ReadBudget > 0 && Socket->HasPendingData(Pending)) {
            int32 Read = 0, Start = Incoming.Num();
            int32 Size = FMath::Min3((int32)Pending, 65536, ReadBudget);
            // TCP may coalesce the end of a maximum-size packet with the next
            // packet. Fill only the current bounded buffer, parse, then read on.
            Size = FMath::Min(Size, Limit + 4 - Start);
            if (Size <= 0) { Fail(TEXT("IPC input too large")); return true; }
            Incoming.AddUninitialized(Size);
            if (!Socket->Recv(Incoming.GetData()+Start, Size, Read)) { Fail(TEXT("IPC read failed")); return true; }
            Incoming.SetNum(Start+Read);
            ReadBudget -= Read;
            // Consume frames promptly so many normal packets cannot exceed the bound.
            while (Incoming.Num() >= 4) {
                uint32 SizeIn = (uint32)Incoming[0] | ((uint32)Incoming[1]<<8) | ((uint32)Incoming[2]<<16) | ((uint32)Incoming[3]<<24);
                if (SizeIn == 0 || SizeIn > Limit) { Fail(TEXT("Invalid IPC packet length")); return true; }
                if (Incoming.Num() < (int32)SizeIn+4) break;
                FUTF8ToTCHAR Convert((const ANSICHAR*)Incoming.GetData()+4, SizeIn);
                if (!Parse(FString(Convert.Length(), Convert.Get()))) { Fail(TEXT("Invalid IPC pose")); return true; }
                Incoming.RemoveAt(0, SizeIn+4, EAllowShrinking::No);
                LastInput = Now;
            }
        }
        if (Now-LastInput > 10 || Socket->GetConnectionState() != SCS_Connected) {
            Fail(TEXT("App connection ended")); return true;
        }
        if (Now-LastPose >= 1./30.) {
            for (const auto& Pair : Actors) Send(Pair.Key, Pair.Value, false);
            LastPose = Now;
        }
        if (Now-LastReference >= .1) {
            for (const auto& Pair : Actors) Send(FName(*(Pair.Key.ToString()+TEXT("_Reference"))), Pair.Value, true);
            LastReference = Now;
        }
        if (Now-LastStatus > .25 && Outgoing.Num() == 0) {
            FString Status = Provider->HasConnection() ? TEXT("{\"connected\":true}\n") : TEXT("{\"connected\":false}\n");
            FTCHARToUTF8 Utf8(*Status); Outgoing.Append((uint8*)Utf8.Get(), Utf8.Length()); LastStatus = Now;
        }
        if (Outgoing.Num()) {
            int32 Written=0;
            Socket->Send(Outgoing.GetData()+OutOffset, Outgoing.Num()-OutOffset, Written);
            OutOffset += Written;
            if (OutOffset == Outgoing.Num()) { Outgoing.Reset(); OutOffset=0; }
        }
        return true;
    }
public:
    virtual void StartupModule() override {
        int32 Port = 0; FString Token;
        if (!FParse::Value(FCommandLine::Get(), TEXT("RGBPort="), Port) || Port < 1 || Port > 65535
            || !FParse::Value(FCommandLine::Get(), TEXT("RGBToken="), Token) || Token.Len() != 64) return;
        auto* Subsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
        Socket = Subsystem->CreateSocket(NAME_Stream, TEXT("RGBMoCap app IPC"), false);
        auto Address = Subsystem->CreateInternetAddr(); bool Valid=false;
        Address->SetIp(TEXT("127.0.0.1"), Valid); Address->SetPort(Port);
        if (!Socket || !Socket->Connect(*Address)) { Fail(TEXT("Cannot connect to app")); return; }
        Socket->SetNoDelay(true);
        FTCHARToUTF8 Utf8(*(Token+TEXT("\n"))); int32 Sent=0;
        if (!Socket->Send((const uint8*)Utf8.Get(), Utf8.Length(), Sent) || Sent != Utf8.Length()) {
            Fail(TEXT("IPC handshake failed")); return;
        }
        Socket->SetNonBlocking(true);
        Provider = ILiveLinkProvider::CreateLiveLinkProvider(TEXT("RGB MoCap"));
        LastInput = FPlatformTime::Seconds();
        Ticker = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateRaw(this, &FRGBMoCapLiveLink::Tick));
    }
    virtual void ShutdownModule() override {
        if (Ticker.IsValid()) FTSTicker::GetCoreTicker().RemoveTicker(Ticker);
        if (Provider.IsValid()) for (const auto& Pair : Actors) Remove(Pair.Key);
        Actors.Empty(); Provider.Reset();
        if (Socket) {
            Socket->Close();
            ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Socket);
            Socket = nullptr;
        }
    }
};
IMPLEMENT_MODULE(FRGBMoCapLiveLink, RGBMoCapLiveLink)
