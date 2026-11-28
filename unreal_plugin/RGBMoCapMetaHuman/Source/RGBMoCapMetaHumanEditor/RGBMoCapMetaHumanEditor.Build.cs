using UnrealBuildTool;
public class RGBMoCapMetaHumanEditor : ModuleRules {
    public RGBMoCapMetaHumanEditor(ReadOnlyTargetRules Target) : base(Target) {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PrivateDependencyModuleNames.AddRange(new string[]{"Core","CoreUObject","Engine","UnrealEd","Slate","SlateCore","InputCore","ToolMenus","LevelEditor","LiveLink","LiveLinkInterface","RGBMoCapMetaHuman","HTTP","ContentBrowser","AssetRegistry"});
    }
}
