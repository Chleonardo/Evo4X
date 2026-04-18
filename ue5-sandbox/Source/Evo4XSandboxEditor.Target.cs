using UnrealBuildTool;

public class Evo4XSandboxEditorTarget : TargetRules
{
	public Evo4XSandboxEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V4;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_4;
		ExtraModuleNames.Add("Evo4XSandbox");
	}
}
