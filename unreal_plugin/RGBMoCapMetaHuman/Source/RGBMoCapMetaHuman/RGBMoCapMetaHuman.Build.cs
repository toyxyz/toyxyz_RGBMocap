using UnrealBuildTool;
public class RGBMoCapMetaHuman : ModuleRules {
    public RGBMoCapMetaHuman(ReadOnlyTargetRules Target) : base(Target) {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[]{"Core","CoreUObject","Engine","LiveLinkInterface"});
        PrivateDependencyModuleNames.AddRange(new string[]{"AnimGraphRuntime","LiveLink"});
    }
}
