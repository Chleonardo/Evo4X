using UnrealBuildTool;

public class Evo4XSandboxTarget : TargetRules
{
	public Evo4XSandboxTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V4;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_4;
		ExtraModuleNames.Add("Evo4XSandbox");
	}
}
