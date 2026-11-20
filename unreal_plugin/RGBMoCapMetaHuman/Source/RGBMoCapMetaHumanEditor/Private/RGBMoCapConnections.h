#pragma once
#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/Views/SListView.h"
#include "Widgets/Input/SCheckBox.h"
#include "RGBMoCapConnectComponent.h"
struct FRGBMoCapSourceRow {
    FName Subject;
    FString Label,State,Detail;
    bool Ready=false,FacePreference=false;
    TArray<TWeakObjectPtr<URGBMoCapConnectComponent>> Connections;
    FString TargetLabels() const;
    FString TargetDetails() const;
    FString ConnectionState() const;
    bool HasMetaHuman() const;
    ECheckBoxState FaceState() const;
};

/** Editor-world connection overview; weak targets and a widget-owned refresh timer. */
class SRGBMoCapConnections : public SCompoundWidget {
public:
    SLATE_BEGIN_ARGS(SRGBMoCapConnections) {} SLATE_END_ARGS()
    void Construct(const FArguments& Args);
private:
    TArray<TSharedPtr<FRGBMoCapSourceRow>> Rows;
    TSharedPtr<SListView<TSharedPtr<FRGBMoCapSourceRow>>> List;
    TSharedPtr<FRGBMoCapSourceRow> Selected;
    FString Message;
    ERGBMoCapTargetType TargetType=ERGBMoCapTargetType::MetaHuman;
    TArray<TSharedPtr<ERGBMoCapTargetType>> TargetTypes;
    EActiveTimerReturnType Poll(double Time,float Delta);
    void UpdateSources();
    TSharedRef<ITableRow> MakeRow(TSharedPtr<FRGBMoCapSourceRow> Row,const TSharedRef<STableViewBase>& Owner);
    void SetFace(ECheckBoxState State,TSharedPtr<FRGBMoCapSourceRow> Row);
    bool CanConnect() const;
    bool CanDisconnect() const;
    FReply Refresh();
    FReply Reconnect();
    FReply Connect();
    FReply Disconnect();
    FReply SelectSource(TSharedPtr<FRGBMoCapSourceRow> Row);
    FReply SelectTargets(TSharedPtr<FRGBMoCapSourceRow> Row);
    FString SelectionHint() const;
};
