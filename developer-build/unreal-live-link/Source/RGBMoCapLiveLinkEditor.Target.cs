using UnrealBuildTool;
public class RGBMoCapLiveLinkEditorTarget : TargetRules {
    public RGBMoCapLiveLinkEditorTarget(TargetInfo Target) : base(Target) {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V6;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_7;
        ExtraModuleNames.Add("RGBMoCapLiveLink");
    }
}
