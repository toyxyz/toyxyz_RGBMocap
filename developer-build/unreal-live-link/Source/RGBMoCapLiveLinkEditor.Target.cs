using UnrealBuildTool;
public class RGBMoCapLiveLinkEditorTarget : TargetRules {
    public RGBMoCapLiveLinkEditorTarget(TargetInfo Target) : base(Target) {
        Type = TargetType.Editor;
        // An editor target of an installed engine shares UnrealEditor's build
        // environment, so it must use that engine's own defaults (UE 5.8
        // rejects V6 warning levels with "modifies the values of properties").
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.Add("RGBMoCapLiveLink");
    }
}
