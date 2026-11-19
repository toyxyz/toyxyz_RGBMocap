using UnrealBuildTool;
public class RGBMoCapLiveLink : ModuleRules {
    public RGBMoCapLiveLink(ReadOnlyTargetRules Target) : base(Target) {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core", "CoreUObject", "Engine", "Json", "Sockets", "Networking",
            "LiveLinkInterface", "LiveLinkMessageBusFramework", "Messaging"
        });
    }
}
