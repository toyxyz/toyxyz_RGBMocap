#line 1 "Source/RGBMoCapMetaHumanEditor/Private/Module.cpp"
#include "Modules/ModuleManager.h"
#include "RGBMoCapConnectComponent.h"
#include "Editor.h"
#include "Engine/Selection.h"
#include "EngineUtils.h"
#include "UObject/ObjectSaveContext.h"
#include "GameFramework/Actor.h"
#include "ScopedTransaction.h"
#include "ToolMenus.h"
#include "Widgets/Docking/SDockTab.h"
#include "Widgets/Input/SButton.h"
#include "RGBMoCapConnections.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Text/STextBlock.h"
#include "Features/IModularFeatures.h"
#include "ILiveLinkClient.h"
#include "LiveLinkMessageBusSourceFactory.h"

class FRGBMoCapEditor : public IModuleInterface {
    FDelegateHandle MenuStartupHandle;
    TSharedRef<SDockTab> Tab(const FSpawnTabArgs&) {
        return SNew(SDockTab).TabRole(ETabRole::NomadTab)[SNew(SRGBMoCapConnections)];
    }
    void PreSave(UWorld* World,FObjectPreSaveContext) {
        for (TActorIterator<AActor> It(World);It;++It) {
            TArray<URGBMoCapConnectComponent*> Components;It->GetComponents(Components);
            for (auto* C:Components) C->SuspendForSave();
        }
    }
    void PostSave(UWorld* World,FObjectPostSaveContext) {
        for (TActorIterator<AActor> It(World);It;++It) {
            TArray<URGBMoCapConnectComponent*> Components;It->GetComponents(Components);
            for (auto* C:Components) C->ResumeAfterSave();
        }
    }
public:
    virtual void StartupModule() override {
        if (IsRunningCommandlet()) return;
        FEditorDelegates::PreSaveWorldWithContext.AddRaw(this,&FRGBMoCapEditor::PreSave);
        FEditorDelegates::PostSaveWorldWithContext.AddRaw(this,&FRGBMoCapEditor::PostSave);
        FGlobalTabmanager::Get()->RegisterNomadTabSpawner(TEXT("RGBMoCapConnect"),FOnSpawnTab::CreateRaw(this,&FRGBMoCapEditor::Tab)).SetDisplayName(FText::FromString(TEXT("RGB MoCap Connect"))).SetMenuType(ETabSpawnerMenuType::Hidden);
        MenuStartupHandle=UToolMenus::RegisterStartupCallback(FSimpleMulticastDelegate::FDelegate::CreateLambda([this](){
            FToolMenuOwnerScoped Owner(this);
            UToolMenus::Get()->ExtendMenu(TEXT("LevelEditor.MainMenu.Tools"))->FindOrAddSection(TEXT("RGBMoCap"))
                .AddMenuEntry(TEXT("RGBMoCapConnect"),FText::FromString(TEXT("RGB MoCap Connect")),FText::FromString(TEXT("Connect a MetaHuman or UE5 mannequin to RGB MoCap Live Link.")),FSlateIcon(),FUIAction(FExecuteAction::CreateLambda([](){FGlobalTabmanager::Get()->TryInvokeTab(FName(TEXT("RGBMoCapConnect")));})));
        }));
    }
    virtual void ShutdownModule() override {
        if (IsRunningCommandlet()) return;
        if (auto Existing=FGlobalTabmanager::Get()->FindExistingLiveTab(FName(TEXT("RGBMoCapConnect")))) Existing->RequestCloseTab();
        FEditorDelegates::PreSaveWorldWithContext.RemoveAll(this);FEditorDelegates::PostSaveWorldWithContext.RemoveAll(this);
        UToolMenus::UnRegisterStartupCallback(MenuStartupHandle);MenuStartupHandle.Reset();
        UToolMenus::UnregisterOwner(this);
        FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(TEXT("RGBMoCapConnect"));
    }
};
IMPLEMENT_MODULE(FRGBMoCapEditor,RGBMoCapMetaHumanEditor)
