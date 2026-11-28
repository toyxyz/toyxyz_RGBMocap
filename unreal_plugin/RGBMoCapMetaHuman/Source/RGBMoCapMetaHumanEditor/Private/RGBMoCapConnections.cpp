#line 1 "Source/RGBMoCapMetaHumanEditor/Private/RGBMoCapConnections.cpp"
#include "RGBMoCapConnections.h"
#include "RGBMoCapBake.h"
#include "Animation/AnimSequence.h"
#include "ContentBrowserModule.h"
#include "IContentBrowserSingleton.h"
#include "RGBMoCapConnectComponent.h"
#include "Editor.h"
#include "Engine/Selection.h"
#include "EngineUtils.h"
#include "Features/IModularFeatures.h"
#include "ILiveLinkClient.h"
#include "Roles/LiveLinkAnimationRole.h"
#include "Roles/LiveLinkAnimationTypes.h"
#include "ScopedTransaction.h"
#include "Styling/AppStyle.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SComboBox.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/SOverlay.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Views/SHeaderRow.h"

namespace {
UWorld* EditorWorld() { return GEditor?GEditor->GetEditorWorldContext().World():nullptr; }
FString SourceLabel(FName Subject) {
    FString Label=Subject.ToString();Label.RemoveFromStart(TEXT("RGBMoCap_"));
    Label.ReplaceInline(TEXT("_"),TEXT(" "));
    return Label.Len()>48?Label.Left(35)+TEXT(" ... ")+Label.Right(8):Label;
}
const TCHAR* TargetTypeLabel(ERGBMoCapTargetType Type) {
    return Type==ERGBMoCapTargetType::MetaHuman?TEXT("MetaHuman"):TEXT("SK Mannequin");
}
AActor* SelectedActor(ERGBMoCapTargetType Type,FString& Reason) {
    if (!GEditor || GEditor->PlayWorld) {Reason=TEXT("Stop Play/Simulate to edit connections.");return nullptr;}
    AActor* Result=nullptr;int32 Count=0;
    for (FSelectionIterator It(*GEditor->GetSelectedActors());It;++It) if (auto* A=Cast<AActor>(*It)) {
        if (IsValid(A) && A->GetWorld()==EditorWorld()) {Result=A;++Count;}
    }
    if (Count!=1) {
        Reason=Count?TEXT("Select only one target actor in the level."):
            FString::Printf(TEXT("Select a %s actor in the level."),TargetTypeLabel(Type));
        return nullptr;
    }
    return URGBMoCapConnectComponent::FindTargetMesh(Result,Type,Reason)?Result:nullptr;
}
ILiveLinkClient* Client() {
    auto& F=IModularFeatures::Get();
    return F.IsModularFeatureAvailable(ILiveLinkClient::ModularFeatureName)?&F.GetModularFeature<ILiveLinkClient>(ILiveLinkClient::ModularFeatureName):nullptr;
}
bool Fresh(const FLiveLinkSubjectFrameData& Frame) {
    const auto* Data=Frame.FrameData.Cast<FLiveLinkAnimationFrameData>();
    if (!Data) return false;
    double At=Data->ArrivalTime.WorldTime>0.?Data->ArrivalTime.WorldTime:Data->WorldTime.GetOffsettedTime();
    double Age=FPlatformTime::Seconds()-At;return Age>=-1. && Age<=2.;
}
void StreamState(ILiveLinkClient* C,FRGBMoCapSourceRow& Row,const TArray<FLiveLinkSubjectKey>& Keys) {
    Row.Ready=false;Row.State=TEXT("Offline");Row.Detail=TEXT("No active stream. Start or reconnect RGB MoCap.");
    if (!C) return;
    TArray<FGuid> Motion,Reference;FName Ref(*(Row.Subject.ToString()+TEXT("_Reference")));
    for (const auto& K:Keys) if (C->IsSubjectEnabled(K,false)) {
        if (K.SubjectName==Row.Subject) Motion.Add(K.Source);
        if (K.SubjectName==Ref) Reference.Add(K.Source);
    }
    if (Motion.Num()>1 || Reference.Num()>1) {Row.State=TEXT("Conflict");Row.Detail=TEXT("Multiple providers use this Subject. Keep one matching motion/reference provider enabled.");return;}
    if (Motion.IsEmpty()) return;
    Row.State=TEXT("Waiting");Row.Detail=TEXT("Waiting for a fresh matching _Reference from the same provider.");
    if (Reference.Num()!=1 || Motion[0]!=Reference[0]) return;
    FLiveLinkSubjectFrameData M,R;
    if (!C->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(Motion[0],Row.Subject),ULiveLinkAnimationRole::StaticClass(),M) || !C->EvaluateFrameFromSource_AnyThread(FLiveLinkSubjectKey(Reference[0],Ref),ULiveLinkAnimationRole::StaticClass(),R)) return;
    if (!Fresh(M) || !Fresh(R)) {Row.State=TEXT("Stale");Row.Detail=TEXT("Stream stopped updating. Connected targets hold their last safe pose.");return;}
    auto* MS=M.StaticData.Cast<FLiveLinkSkeletonStaticData>();auto* RS=R.StaticData.Cast<FLiveLinkSkeletonStaticData>();
    if (!MS || !RS || MS->BoneNames!=RS->BoneNames || MS->BoneParents!=RS->BoneParents) {Row.Detail=TEXT("Motion and reference skeletons do not match.");return;}
    Row.Ready=true;Row.State=TEXT("Live");Row.Detail=TEXT("Fresh motion and matching reference received. Ready to connect.");
}
class SRGBMoCapSourceTableRow : public SMultiColumnTableRow<TSharedPtr<FRGBMoCapSourceRow>> {
public:
    SLATE_BEGIN_ARGS(SRGBMoCapSourceTableRow) {} SLATE_ARGUMENT(TSharedPtr<FRGBMoCapSourceRow>, Item) SLATE_EVENT(FOnClicked, OnSelectTargets) SLATE_EVENT(FOnClicked, OnSelectSource) SLATE_EVENT(FOnCheckStateChanged, OnFace) SLATE_ATTRIBUTE(bool, FaceAvailable) SLATE_END_ARGS()
    void Construct(const FArguments& Args,const TSharedRef<STableViewBase>& Owner) {
        Item=Args._Item;OnSelectTargets=Args._OnSelectTargets;OnSelectSource=Args._OnSelectSource;OnFace=Args._OnFace;FaceAvailable=Args._FaceAvailable;
        SMultiColumnTableRow::Construct(FSuperRowType::FArguments().SignalSelectionMode(ETableRowSignalSelectionMode::Instantaneous).Padding(FMargin(4,7)).Tag(Item->Subject),Owner);
    }
    TSharedRef<SWidget> GenerateWidgetForColumn(const FName& Column) override {
        if (Column==TEXT("Face")) return SNew(SCheckBox).Tag(FName(*(TEXT("Face_")+Item->Subject.ToString())))
            .IsChecked_Lambda([this](){return FaceAvailable.Get()?Item->FaceState():ECheckBoxState::Unchecked;})
            .OnCheckStateChanged(OnFace)
            .IsEnabled_Lambda([this](){return FaceAvailable.Get() && GEditor && !GEditor->PlayWorld;})
            .ToolTipText(FText::FromString(TEXT("Apply tracked ARKit expressions to MetaHumans on this source. SK Mannequin uses body motion only.")));
        if (Column==TEXT("Source")) return SNew(SButton).ButtonStyle(FAppStyle::Get(),"SimpleButton").ContentPadding(0).HAlign(HAlign_Left)
            .Tag(FName(*(TEXT("Source_")+Item->Subject.ToString()))).ToolTipText(FText::FromName(Item->Subject)).OnClicked(OnSelectSource)
            [SNew(STextBlock).Text_Lambda([R=Item](){return FText::FromString(R->Label);})];
        if (Column==TEXT("State")) return SNew(STextBlock).Text_Lambda([R=Item](){return FText::FromString(R->State);})
            .ColorAndOpacity_Lambda([R=Item](){return R->Ready?FSlateColor(FLinearColor(.3f,.8f,.4f)):FSlateColor::UseSubduedForeground();})
            .ToolTipText_Lambda([R=Item](){return FText::FromString(R->Detail);});
        return SNew(SVerticalBox)
            +SVerticalBox::Slot().AutoHeight()[SNew(SButton).ButtonStyle(FAppStyle::Get(),"SimpleButton").HAlign(HAlign_Left).ContentPadding(FMargin(0))
            .Tag(FName(*(TEXT("Targets_")+Item->Subject.ToString())))
            .IsEnabled_Lambda([R=Item](){return !R->Connections.IsEmpty();})
            .ToolTipText_Lambda([R=Item](){return FText::FromString(R->TargetDetails());}).OnClicked(OnSelectTargets)
            [SNew(STextBlock).Text_Lambda([R=Item](){return FText::FromString(R->TargetLabels());}).AutoWrapText(true)]]
            +SVerticalBox::Slot().AutoHeight()[SNew(STextBlock).Text_Lambda([R=Item](){return FText::FromString(R->ConnectionState());}).AutoWrapText(true)
                .ColorAndOpacity(FSlateColor::UseSubduedForeground()).Visibility_Lambda([R=Item](){return R->Connections.IsEmpty()?EVisibility::Collapsed:EVisibility::Visible;})];
    }
private:
    TSharedPtr<FRGBMoCapSourceRow> Item;FOnClicked OnSelectTargets,OnSelectSource;FOnCheckStateChanged OnFace;
    TAttribute<bool> FaceAvailable;
};
}
FString FRGBMoCapSourceRow::TargetLabels() const {
    TArray<FString> Labels;
    for (const auto& W:Connections) if (auto* C=W.Get()) if (IsValid(C->GetOwner())) Labels.Add(FString::Printf(TEXT("%s (%s)"),*C->GetOwner()->GetActorLabel(),TargetTypeLabel(C->TargetType)));
    return Labels.IsEmpty()?TEXT("Not connected"):FString::Join(Labels,TEXT(", "));
}
FString FRGBMoCapSourceRow::TargetDetails() const {
    TArray<FString> Lines;
    for (const auto& W:Connections) if (auto* C=W.Get()) if (auto* A=C->GetOwner()) Lines.Add(A->GetActorLabel()+TEXT(" (")+TargetTypeLabel(C->TargetType)+TEXT(")\n")+A->GetPathName()+TEXT("\n")+C->Status);
    return Lines.IsEmpty()?TEXT("Select this source and a level actor of the chosen target type, then Connect selected actor."):FString::Join(Lines,TEXT("\n\n"))+TEXT("\n\nClick to select connected actors in the level.");
}
FString FRGBMoCapSourceRow::ConnectionState() const {
    TArray<FString> States;
    for (const auto& W:Connections) if (auto* C=W.Get()) {
        FString ConnectionStatus=C->Status;
        if (ConnectionStatus.StartsWith(TEXT("Streaming - direct body"))) ConnectionStatus=TEXT("Streaming");
        else if (ConnectionStatus.StartsWith(TEXT("Streaming - foot reach"))) ConnectionStatus=TEXT("Streaming - foot reach limited");
        else if (ConnectionStatus.StartsWith(TEXT("Connected. Waiting"))) ConnectionStatus=TEXT("Waiting for motion");
        if (C->SupportsFace() && C->FaceEnabled) ConnectionStatus+=TEXT("; Face: ")+C->FaceStatus;
        States.AddUnique(ConnectionStatus);
    }
    return FString::Join(States,TEXT("; "));
}
bool FRGBMoCapSourceRow::HasMetaHuman() const {
    return Connections.ContainsByPredicate([](const auto& W) {
        auto* C=W.Get();return C && C->SupportsFace();
    });
}
ECheckBoxState FRGBMoCapSourceRow::FaceState() const {
    int32 Enabled=0,Total=0;
    for (const auto& W:Connections) if (auto* C=W.Get(); C && C->SupportsFace()) {++Total;if (C->FaceEnabled) ++Enabled;}
    if (!Total) return FacePreference?ECheckBoxState::Checked:ECheckBoxState::Unchecked;
    return Enabled==Total?ECheckBoxState::Checked:(Enabled?ECheckBoxState::Undetermined:ECheckBoxState::Unchecked);
}
void SRGBMoCapConnections::SetFace(ECheckBoxState State,TSharedPtr<FRGBMoCapSourceRow> Row) {
    if (!GEditor || GEditor->PlayWorld || (TargetType==ERGBMoCapTargetType::Mannequin && !Row->HasMetaHuman())) return;
    bool Enable=State==ECheckBoxState::Checked;Row->FacePreference=Enable;
    FScopedTransaction Transaction(FText::FromString(TEXT("Change RGB MoCap face streaming")));
    for (const auto& W:Row->Connections) if (auto* C=W.Get(); C && C->SupportsFace()) {C->GetOwner()->Modify();C->SetFaceEnabled(Enable);}
}
void SRGBMoCapConnections::Construct(const FArguments&) {
    URGBMoCapConnectComponent::FindAppSource();UpdateSources();
    TargetTypes={MakeShared<ERGBMoCapTargetType>(ERGBMoCapTargetType::MetaHuman),
                 MakeShared<ERGBMoCapTargetType>(ERGBMoCapTargetType::Mannequin)};
    ChildSlot[SNew(SBox).MinDesiredWidth(720).MinDesiredHeight(350).Padding(12)
    [SNew(SVerticalBox)
        +SVerticalBox::Slot().AutoHeight().Padding(0,0,0,8)
        [SNew(SHorizontalBox)
            +SHorizontalBox::Slot().FillWidth(1).VAlign(VAlign_Center)[SNew(STextBlock).Text(FText::FromString(TEXT("Live Link sources and targets")))]
            +SHorizontalBox::Slot().AutoWidth().Padding(4,0)[SNew(SButton).Text(FText::FromString(TEXT("Refresh"))).OnClicked(this,&SRGBMoCapConnections::Refresh)]
            +SHorizontalBox::Slot().AutoWidth()[SNew(SButton).Text(FText::FromString(TEXT("Reconnect Live Link"))).OnClicked(this,&SRGBMoCapConnections::Reconnect)]]
        +SVerticalBox::Slot().FillHeight(1)
        [SNew(SOverlay)
            +SOverlay::Slot()[SAssignNew(List,SListView<TSharedPtr<FRGBMoCapSourceRow>>).ListItemsSource(&Rows).SelectionMode(ESelectionMode::Single)
                .OnGenerateRow(this,&SRGBMoCapConnections::MakeRow)
                .OnSelectionChanged_Lambda([this](TSharedPtr<FRGBMoCapSourceRow> Row,ESelectInfo::Type){Selected=Row;Message.Reset();})
                .HeaderRow(SNew(SHeaderRow)
                    +SHeaderRow::Column(TEXT("Source")).DefaultLabel(FText::FromString(TEXT("Source"))).FillWidth(.35f)
                    +SHeaderRow::Column(TEXT("State")).DefaultLabel(FText::FromString(TEXT("Stream"))).FixedWidth(85)
                    +SHeaderRow::Column(TEXT("Targets")).DefaultLabel(FText::FromString(TEXT("Connected targets"))).FillWidth(.65f)
                    +SHeaderRow::Column(TEXT("Face")).DefaultLabel(FText::FromString(TEXT("Face"))).FixedWidth(45))]
            +SOverlay::Slot().HAlign(HAlign_Center).VAlign(VAlign_Center)
            [SNew(STextBlock).Text(FText::FromString(TEXT("No RGB MoCap sources. Start streaming in the app."))).Visibility_Lambda([this](){return Rows.IsEmpty()?EVisibility::HitTestInvisible:EVisibility::Collapsed;})]]
        +SVerticalBox::Slot().AutoHeight().Padding(0,10,0,7)[SNew(SSeparator)]
        +SVerticalBox::Slot().AutoHeight().Padding(0,0,0,5)
        [SNew(STextBlock).Text_Lambda([this](){return FText::FromString(TEXT("Selected source: ")+(Selected?Selected->Label:TEXT("Select a row above")));})]
        +SVerticalBox::Slot().AutoHeight().Padding(0,0,0,8)
        [SNew(SHorizontalBox)
            +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,6,0)
            [SNew(STextBlock).Text(FText::FromString(TEXT("Target type")))]
            +SHorizontalBox::Slot().AutoWidth().Padding(0,0,12,0)
            [SNew(SComboBox<TSharedPtr<ERGBMoCapTargetType>>).Tag(TEXT("RGBMoCapTargetType"))
                .OptionsSource(&TargetTypes).InitiallySelectedItem(TargetTypes[0])
                .IsEnabled_Lambda([](){return GEditor && !GEditor->PlayWorld;})
                .OnGenerateWidget_Lambda([](TSharedPtr<ERGBMoCapTargetType> Item){
                    const TCHAR* Label=*Item==ERGBMoCapTargetType::MetaHuman?TEXT("MetaHuman"):TEXT("SK Mannequin (Manny/Quinn)");
                    return SNew(STextBlock).Text(FText::FromString(Label));
                })
                .OnSelectionChanged_Lambda([this](TSharedPtr<ERGBMoCapTargetType> Item,ESelectInfo::Type){
                    if (Item) {TargetType=*Item;Message.Reset();}
                })
                [SNew(STextBlock).Text_Lambda([this](){return FText::FromString(TargetTypeLabel(TargetType));})]]
            +SHorizontalBox::Slot().FillWidth(1).VAlign(VAlign_Center)
            [SNew(STextBlock).Tag(TEXT("RGBMoCapTargetSelection")).Text_Lambda([this](){return FText::FromString(SelectionHint());}).AutoWrapText(true)]]
        +SVerticalBox::Slot().AutoHeight()
        [SNew(SHorizontalBox)
            +SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)[SNew(SButton).Tag(TEXT("RGBMoCapConnectActor")).Text(FText::FromString(TEXT("Connect selected actor"))).IsEnabled(this,&SRGBMoCapConnections::CanConnect).OnClicked(this,&SRGBMoCapConnections::Connect)]
            +SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)
            [SNew(SButton).Tag(TEXT("RGBMoCapBake")).Text(FText::FromString(TEXT("Bake...")))
                .ToolTipText(FText::FromString(TEXT("Save the connected take as an Animation Sequence. If several actors are connected, select one in the level.")))
                .IsEnabled(this,&SRGBMoCapConnections::CanBake).OnClicked(this,&SRGBMoCapConnections::Bake)]
            +SHorizontalBox::Slot().AutoWidth()[SNew(SButton).Tag(TEXT("RGBMoCapDisconnectSource")).Text(FText::FromString(TEXT("Disconnect source"))).ToolTipText(FText::FromString(TEXT("Disconnect all actors listed on the selected source row and restore their animation."))).IsEnabled(this,&SRGBMoCapConnections::CanDisconnect).OnClicked(this,&SRGBMoCapConnections::Disconnect)]]
        +SVerticalBox::Slot().AutoHeight().Padding(0,8,0,0)[SNew(STextBlock).Text_Lambda([this](){return FText::FromString(Message);}).AutoWrapText(true)]
    ]];
    if (Selected) List->SetSelection(Selected);
    RegisterActiveTimer(.5f,FWidgetActiveTimerDelegate::CreateSP(this,&SRGBMoCapConnections::Poll));
}
EActiveTimerReturnType SRGBMoCapConnections::Poll(double,float) {UpdateSources();return EActiveTimerReturnType::Continue;}
void SRGBMoCapConnections::UpdateSources() {
    TMap<FName,TArray<TWeakObjectPtr<URGBMoCapConnectComponent>>> Bound;
    if (auto* World=EditorWorld()) for (TActorIterator<AActor> It(World);It;++It) {
        if (!IsValid(*It) || It->IsTemplate() || It->IsActorBeingDestroyed()) continue;
        TArray<URGBMoCapConnectComponent*> Components;It->GetComponents(Components);
        for (auto* C:Components) if (IsValid(C) && C->Enabled && !C->Subject.IsNone() && !C->Subject.ToString().EndsWith(TEXT("_Reference"))) Bound.FindOrAdd(C->Subject).Add(C);
    }
    ILiveLinkClient* C=Client();TArray<FLiveLinkSubjectKey> Keys=C?C->GetSubjectsSupportingRole(ULiveLinkAnimationRole::StaticClass(),true,false):TArray<FLiveLinkSubjectKey>();
    TArray<FName> Names;Bound.GetKeys(Names);
    for (const auto& K:Keys) {
        FString Name=K.SubjectName.ToString();if (Name.StartsWith(TEXT("RGBMoCap_")) && !Name.EndsWith(TEXT("_Reference")) && !Name.EndsWith(TEXT("_Face"))) Names.AddUnique(K.SubjectName.Name);
    }
    Names.Sort([](FName A,FName B){return A.LexicalLess(B);});
    TArray<TSharedPtr<FRGBMoCapSourceRow>> Next;
    for (FName Name:Names) {
        auto* Existing=Rows.FindByPredicate([&](const auto& R){return R->Subject==Name;});
        auto Row=Existing?*Existing:MakeShared<FRGBMoCapSourceRow>();Row->Subject=Name;Row->Label=SourceLabel(Name);
        Row->Connections=Bound.FindRef(Name);
        Row->Connections.Sort([](const auto& A,const auto& B){return A.IsValid() && B.IsValid()?A->GetOwner()->GetPathName()<B->GetOwner()->GetPathName():A.IsValid();});
        StreamState(C,*Row,Keys);Next.Add(Row);
    }
    bool Changed=Next!=Rows;Rows=MoveTemp(Next);
    if (Selected && !Rows.Contains(Selected)) {Selected.Reset();if (List) List->ClearSelection();}

    if (List && Changed) List->RequestListRefresh();
}
TSharedRef<ITableRow> SRGBMoCapConnections::MakeRow(TSharedPtr<FRGBMoCapSourceRow> Row,const TSharedRef<STableViewBase>& Owner) {
    return SNew(SRGBMoCapSourceTableRow,Owner).Tag(Row->Subject).Item(Row)
        .FaceAvailable_Lambda([this,Row](){return TargetType==ERGBMoCapTargetType::MetaHuman || Row->HasMetaHuman();}).OnFace(FOnCheckStateChanged::CreateSP(this,&SRGBMoCapConnections::SetFace,Row)).OnSelectSource(FOnClicked::CreateSP(this,&SRGBMoCapConnections::SelectSource,Row)).OnSelectTargets(FOnClicked::CreateSP(this,&SRGBMoCapConnections::SelectTargets,Row));
}
bool SRGBMoCapConnections::CanConnect() const {FString Why;return Selected && Selected->Ready && SelectedActor(TargetType,Why);}
bool SRGBMoCapConnections::CanDisconnect() const {return Selected && !Selected->Connections.IsEmpty() && GEditor && !GEditor->PlayWorld;}
FString SRGBMoCapConnections::SelectionHint() const {
    FString Why;auto* A=SelectedActor(TargetType,Why);
    return A?TEXT("Selected level actor: ")+A->GetActorLabel():Why;
}
FReply SRGBMoCapConnections::Refresh() {URGBMoCapConnectComponent::FindAppSource();UpdateSources();return FReply::Handled();}
FReply SRGBMoCapConnections::Reconnect() {URGBMoCapConnectComponent::ReconnectAppSource();UpdateSources();Message=TEXT("Requested Live Link reconnection.");return FReply::Handled();}
FReply SRGBMoCapConnections::Connect() {
    UpdateSources();FString Why;auto* A=SelectedActor(TargetType,Why);
    if (!Selected || !Selected->Ready || !A) {Message=A?TEXT("Select a live source with a matching reference."):Why;return FReply::Handled();}
    FScopedTransaction Transaction(FText::FromString(TEXT("Connect RGB MoCap target")));
    const bool EnableFace=TargetType==ERGBMoCapTargetType::MetaHuman && Selected->FaceState()==ECheckBoxState::Checked;
    auto* Component=URGBMoCapConnectComponent::ConnectActorForTarget(A,Selected->Subject,TargetType);
    if (Component) Component->SetFaceEnabled(EnableFace);
    Message=Component && Component->Enabled?TEXT("Connected ")+Selected->Label+TEXT(" to ")+A->GetActorLabel()+TEXT("."):(Component?Component->Status:TEXT("Could not connect the selected actor."));
    UpdateSources();GEditor->RedrawLevelEditingViewports();return FReply::Handled();
}
URGBMoCapConnectComponent* SRGBMoCapConnections::BakeTarget() const {
    if (!Selected || !Selected->Ready || !GEditor || GEditor->PlayWorld) return nullptr;
    URGBMoCapConnectComponent* Only=nullptr;
    URGBMoCapConnectComponent* Chosen=nullptr;
    int32 Count=0,SelectedCount=0;
    for (const auto& Weak:Selected->Connections) if (auto* C=Weak.Get();C && C->Enabled && IsValid(C->GetOwner())) {
        Only=C;++Count;
        if (C->GetOwner()->IsSelectedInEditor()) {Chosen=C;++SelectedCount;}
    }
    return Count==1?Only:SelectedCount==1?Chosen:nullptr;
}

bool SRGBMoCapConnections::CanBake() const {return !Baking && BakeTarget()!=nullptr;}

FReply SRGBMoCapConnections::Bake() {
    auto* Target=BakeTarget();
    if (!Target || Baking) return FReply::Handled();
    FRGBMoCapBakeContext Context;
    if (!Target->CaptureBake(Context,Message)) return FReply::Handled();
    const auto Connection=TWeakObjectPtr<URGBMoCapConnectComponent>(Target);
    FSaveAssetDialogConfig Config;
    Config.DialogTitleOverride=FText::FromString(TEXT("Save Retargeted Animation"));
    Config.DefaultPath=TEXT("/Game");
    Config.DefaultAssetName=Context.Subject.ToString()+TEXT("_Animation");
    Config.AssetClassNames.Add(UAnimSequence::StaticClass()->GetClassPathName());
    auto& Browser=FModuleManager::LoadModuleChecked<FContentBrowserModule>(TEXT("ContentBrowser")).Get();
    const FString Path=Browser.CreateModalSaveAssetDialog(Config);
    if (Path.IsEmpty()) return FReply::Handled();
    if (!Connection.IsValid() || !Connection->MatchesBakeTarget(Context)) {
        Message=TEXT("The target connection changed. Select Bake again.");return FReply::Handled();
    }
    Message=Context.FaceRequested?TEXT("Baking the connected motion and available face data..."):
        TEXT("Baking the connected body motion...");
    Baking=true;
    const TWeakPtr<SRGBMoCapConnections> Widget=SharedThis(this);
    RGBMoCap::RequestBake(Context,[Widget,Connection,Context,Path](RGBMoCap::FBakedTracks Tracks,FString Error) {
        const auto Self=Widget.Pin();
        if (!Self) return;
        Self->Baking=false;
        if (!Error.IsEmpty()) {Self->Message=MoveTemp(Error);return;}
        if (!GEditor || GEditor->PlayWorld || !Connection.IsValid() || !Connection->MatchesBakeTarget(Context)) {
            Self->Message=TEXT("The target connection changed before the bake completed.");return;
        }
        FString FacePath;
        if (!Tracks.FaceKeys.IsEmpty()) {
            if (Connection->MatchesFaceBakeTarget(Context)) {
                FacePath=RGBMoCap::FaceBakePath(Path);
                FString Filename;
                if (!RGBMoCap::CheckBakeDestination(FacePath,Filename,Error)) {
                    Self->Message=MoveTemp(Error);return;
                }
            } else {
                Tracks.FaceWarning=TEXT("The Face target or setting changed during the bake.");
            }
        }
        auto* Asset=RGBMoCap::SaveBake(Context,Tracks,Path,Error);
        if (!Asset) {Self->Message=MoveTemp(Error);return;}
        Self->Message=FString::Printf(TEXT("Saved %s (%d samples)."),*Path,Tracks.Frames);
        TArray<UObject*> Assets{Asset};
        if (!FacePath.IsEmpty()) {
            if (auto* Face=RGBMoCap::SaveBake(Context,Tracks,FacePath,Error,true)) {
                Assets.Add(Face);
                Self->Message+=TEXT(" Face saved to ")+FacePath+TEXT(".");
            } else {
                Tracks.FaceWarning=MoveTemp(Error);
            }
        }
        if (!Tracks.FaceWarning.IsEmpty()) Self->Message+=TEXT(" Face skipped: ")+Tracks.FaceWarning;
        FModuleManager::LoadModuleChecked<FContentBrowserModule>(TEXT("ContentBrowser")).Get().SyncBrowserToAssets(Assets);
    });
    return FReply::Handled();
}

FReply SRGBMoCapConnections::Disconnect() {
    UpdateSources();if (!CanDisconnect()) return FReply::Handled();
    FScopedTransaction Transaction(FText::FromString(TEXT("Disconnect RGB MoCap source")));
    auto Connections=Selected->Connections;
    for (const auto& W:Connections) if (auto* C=W.Get()) {C->GetOwner()->Modify();C->Modify();C->Disconnect();}
    Message=TEXT("Disconnected ")+Selected->Label+TEXT(". Original animation restored.");
    UpdateSources();GEditor->RedrawLevelEditingViewports();return FReply::Handled();
}
FReply SRGBMoCapConnections::SelectSource(TSharedPtr<FRGBMoCapSourceRow> Row) {
    Selected=Row;if (List) List->SetSelection(Row);Message.Reset();return FReply::Handled();
}
FReply SRGBMoCapConnections::SelectTargets(TSharedPtr<FRGBMoCapSourceRow> Row) {
    if (!GEditor || GEditor->PlayWorld) return FReply::Handled();
    GEditor->SelectNone(false,true);
    for (const auto& W:Row->Connections) if (auto* C=W.Get()) if (IsValid(C->GetOwner())) GEditor->SelectActor(C->GetOwner(),true,false);
    GEditor->NoteSelectionChange();return FReply::Handled();
}
