#line 1 "Source/RGBMoCapMetaHumanEditor/Private/RGBMoCapConnections.cpp"
#include "RGBMoCapConnections.h"
#include "RGBMoCapConnectComponent.h"
#include "Editor.h"
#include "Engine/Selection.h"
#include "Engine/SkeletalMesh.h"
#include "EngineUtils.h"
#include "Features/IModularFeatures.h"
#include "ILiveLinkClient.h"
#include "Roles/LiveLinkAnimationRole.h"
#include "Roles/LiveLinkAnimationTypes.h"
#include "ScopedTransaction.h"
#include "Styling/AppStyle.h"
#include "Widgets/Input/SButton.h"
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
AActor* SelectedActor(FString& Reason) {
    if (!GEditor || GEditor->PlayWorld) {Reason=TEXT("Stop Play/Simulate to edit connections.");return nullptr;}
    AActor* Result=nullptr;int32 Count=0;
    for (FSelectionIterator It(*GEditor->GetSelectedActors());It;++It) if (auto* A=Cast<AActor>(*It)) {
        if (IsValid(A) && A->GetWorld()==EditorWorld()) {Result=A;++Count;}
    }
    if (Count!=1) {Reason=Count?TEXT("Select only one MetaHuman actor in the level."):TEXT("Select a MetaHuman actor in the level.");return nullptr;}
    USkeletalMeshComponent* Body=nullptr;
    TArray<USkeletalMeshComponent*> Meshes;Result->GetComponents(Meshes);
    for (auto* M:Meshes) if (M->GetFName()==TEXT("Body")) Body=M;
    if (!Body || !Body->GetSkeletalMeshAsset()) {Reason=TEXT("Selected actor needs a MetaHuman Body skeletal mesh.");return nullptr;}
    const auto& Ref=Body->GetSkeletalMeshAsset()->GetRefSkeleton();
    for (const TCHAR* Name:{TEXT("pelvis"),TEXT("thigh_l"),TEXT("calf_l"),TEXT("foot_l"),TEXT("ball_l"),TEXT("thigh_r"),TEXT("calf_r"),TEXT("foot_r"),TEXT("ball_r")})
        if (Ref.FindBoneIndex(Name)<0) {Reason=FString::Printf(TEXT("Selected Body is missing %s."),Name);return nullptr;}
    if (Body->GetAnimationMode()!=EAnimationMode::AnimationBlueprint) {Reason=TEXT("Selected Body must use Animation Blueprint mode.");return nullptr;}
    auto Scale=Body->GetComponentScale();
    if (Scale.X<=0. || !Scale.AllComponentsEqual(1.e-4)) {Reason=TEXT("Selected Body needs positive uniform scale.");return nullptr;}
    Reason=Result->GetActorLabel();return Result;
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
    SLATE_BEGIN_ARGS(SRGBMoCapSourceTableRow) {} SLATE_ARGUMENT(TSharedPtr<FRGBMoCapSourceRow>, Item) SLATE_EVENT(FOnClicked, OnSelectTargets) SLATE_EVENT(FOnClicked, OnSelectSource) SLATE_EVENT(FOnCheckStateChanged, OnFace) SLATE_END_ARGS()
    void Construct(const FArguments& Args,const TSharedRef<STableViewBase>& Owner) {
        Item=Args._Item;OnSelectTargets=Args._OnSelectTargets;OnSelectSource=Args._OnSelectSource;OnFace=Args._OnFace;
        SMultiColumnTableRow::Construct(FSuperRowType::FArguments().SignalSelectionMode(ETableRowSignalSelectionMode::Instantaneous).Padding(FMargin(4,7)).Tag(Item->Subject),Owner);
    }
    TSharedRef<SWidget> GenerateWidgetForColumn(const FName& Column) override {
        if (Column==TEXT("Face")) return SNew(SCheckBox).Tag(FName(*(TEXT("Face_")+Item->Subject.ToString())))
            .IsChecked_Lambda([R=Item](){return R->FaceState();}).OnCheckStateChanged(OnFace)
            .IsEnabled_Lambda([](){return GEditor && !GEditor->PlayWorld;})
            .ToolTipText(FText::FromString(TEXT("Enable tracked ARKit facial expressions for the actors on this source. Waits safely when no Track Face data is available.")));
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
};
}
FString FRGBMoCapSourceRow::TargetLabels() const {
    TArray<FString> Labels;
    for (const auto& W:Connections) if (auto* C=W.Get()) if (IsValid(C->GetOwner())) Labels.Add(C->GetOwner()->GetActorLabel());
    return Labels.IsEmpty()?TEXT("Not connected"):FString::Join(Labels,TEXT(", "));
}
FString FRGBMoCapSourceRow::TargetDetails() const {
    TArray<FString> Lines;
    for (const auto& W:Connections) if (auto* C=W.Get()) if (auto* A=C->GetOwner()) Lines.Add(A->GetActorLabel()+TEXT("\n")+A->GetPathName()+TEXT("\n")+C->Status);
    return Lines.IsEmpty()?TEXT("Select this source and a MetaHuman actor, then Connect selected actor."):FString::Join(Lines,TEXT("\n\n"))+TEXT("\n\nClick to select connected actors in the level.");
}
FString FRGBMoCapSourceRow::ConnectionState() const {
    TArray<FString> States;
    for (const auto& W:Connections) if (auto* C=W.Get()) {
        FString ConnectionStatus=C->Status;
        if (ConnectionStatus.StartsWith(TEXT("Streaming - direct body"))) ConnectionStatus=TEXT("Streaming");
        else if (ConnectionStatus.StartsWith(TEXT("Streaming - foot reach"))) ConnectionStatus=TEXT("Streaming - foot reach limited");
        else if (ConnectionStatus.StartsWith(TEXT("Connected. Waiting"))) ConnectionStatus=TEXT("Waiting for motion");
        if (C->FaceEnabled) ConnectionStatus+=TEXT("; Face: ")+C->FaceStatus;
        States.AddUnique(ConnectionStatus);
    }
    return FString::Join(States,TEXT("; "));
}
ECheckBoxState FRGBMoCapSourceRow::FaceState() const {
    int32 Enabled=0,Total=0;
    for (const auto& W:Connections) if (auto* C=W.Get()) {++Total;if (C->FaceEnabled) ++Enabled;}
    if (!Total) return FacePreference?ECheckBoxState::Checked:ECheckBoxState::Unchecked;
    return Enabled==Total?ECheckBoxState::Checked:(Enabled?ECheckBoxState::Undetermined:ECheckBoxState::Unchecked);
}
void SRGBMoCapConnections::SetFace(ECheckBoxState State,TSharedPtr<FRGBMoCapSourceRow> Row) {
    if (!GEditor || GEditor->PlayWorld) return;
    bool Enable=State==ECheckBoxState::Checked;Row->FacePreference=Enable;
    FScopedTransaction Transaction(FText::FromString(TEXT("Change RGB MoCap face streaming")));
    for (const auto& W:Row->Connections) if (auto* C=W.Get()) {C->GetOwner()->Modify();C->SetFaceEnabled(Enable);}
}
void SRGBMoCapConnections::Construct(const FArguments&) {
    URGBMoCapConnectComponent::FindAppSource();UpdateSources();
    ChildSlot[SNew(SBox).MinDesiredWidth(720).MinDesiredHeight(350).Padding(12)
    [SNew(SVerticalBox)
        +SVerticalBox::Slot().AutoHeight().Padding(0,0,0,8)
        [SNew(SHorizontalBox)
            +SHorizontalBox::Slot().FillWidth(1).VAlign(VAlign_Center)[SNew(STextBlock).Text(FText::FromString(TEXT("Live Link sources and MetaHuman connections")))]
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
                    +SHeaderRow::Column(TEXT("Targets")).DefaultLabel(FText::FromString(TEXT("Connected MetaHuman"))).FillWidth(.65f)
                    +SHeaderRow::Column(TEXT("Face")).DefaultLabel(FText::FromString(TEXT("Face"))).FixedWidth(45))]
            +SOverlay::Slot().HAlign(HAlign_Center).VAlign(VAlign_Center)
            [SNew(STextBlock).Text(FText::FromString(TEXT("No RGB MoCap sources. Start streaming in the app."))).Visibility_Lambda([this](){return Rows.IsEmpty()?EVisibility::HitTestInvisible:EVisibility::Collapsed;})]]
        +SVerticalBox::Slot().AutoHeight().Padding(0,10,0,7)[SNew(SSeparator)]
        +SVerticalBox::Slot().AutoHeight().Padding(0,0,0,5)
        [SNew(STextBlock).Text_Lambda([this](){return FText::FromString(TEXT("Selected source: ")+(Selected?Selected->Label:TEXT("Select a row above")));})]
        +SVerticalBox::Slot().AutoHeight().Padding(0,0,0,8)
        [SNew(STextBlock).Tag(TEXT("RGBMoCapTargetSelection")).Text_Lambda([this](){return FText::FromString(SelectionHint());}).AutoWrapText(true)]
        +SVerticalBox::Slot().AutoHeight()
        [SNew(SHorizontalBox)
            +SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)[SNew(SButton).Tag(TEXT("RGBMoCapConnectActor")).Text(FText::FromString(TEXT("Connect selected actor"))).IsEnabled(this,&SRGBMoCapConnections::CanConnect).OnClicked(this,&SRGBMoCapConnections::Connect)]
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
    return SNew(SRGBMoCapSourceTableRow,Owner).Tag(Row->Subject).Item(Row).OnFace(FOnCheckStateChanged::CreateSP(this,&SRGBMoCapConnections::SetFace,Row)).OnSelectSource(FOnClicked::CreateSP(this,&SRGBMoCapConnections::SelectSource,Row)).OnSelectTargets(FOnClicked::CreateSP(this,&SRGBMoCapConnections::SelectTargets,Row));
}
bool SRGBMoCapConnections::CanConnect() const {FString Why;return Selected && Selected->Ready && SelectedActor(Why);}
bool SRGBMoCapConnections::CanDisconnect() const {return Selected && !Selected->Connections.IsEmpty() && GEditor && !GEditor->PlayWorld;}
FString SRGBMoCapConnections::SelectionHint() const {
    FString Why;auto* A=SelectedActor(Why);
    return A?TEXT("Selected level actor: ")+A->GetActorLabel():Why;
}
FReply SRGBMoCapConnections::Refresh() {URGBMoCapConnectComponent::FindAppSource();UpdateSources();return FReply::Handled();}
FReply SRGBMoCapConnections::Reconnect() {URGBMoCapConnectComponent::ReconnectAppSource();UpdateSources();Message=TEXT("Requested Live Link reconnection.");return FReply::Handled();}
FReply SRGBMoCapConnections::Connect() {
    UpdateSources();FString Why;auto* A=SelectedActor(Why);
    if (!Selected || !Selected->Ready || !A) {Message=A?TEXT("Select a live source with a matching reference."):Why;return FReply::Handled();}
    FScopedTransaction Transaction(FText::FromString(TEXT("Connect RGB MoCap MetaHuman")));
    bool EnableFace=Selected->FaceState()==ECheckBoxState::Checked;
    auto* Component=URGBMoCapConnectComponent::ConnectActor(A,Selected->Subject);
    if (Component) Component->SetFaceEnabled(EnableFace);
    Message=Component && Component->Enabled?TEXT("Connected ")+Selected->Label+TEXT(" to ")+A->GetActorLabel()+TEXT("."):(Component?Component->Status:TEXT("Could not connect the selected actor."));
    UpdateSources();GEditor->RedrawLevelEditingViewports();return FReply::Handled();
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
