#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "SimCore/WorldState.h"
#include "SimCore/SpeciesData.h"
#include "SimCore/BiomeData.h"
#include "SandboxHUD.generated.h"

/**
 * HUD for the sandbox — draws all UI panels directly on screen.
 * No UMG dependency for v1 — pure Canvas drawing for simplicity.
 *
 * Panels:
 * - Global Panel (top-left): tick, speed, biomass, species count
 * - Region Inspector (right side): details of selected region
 * - Controls hint (bottom): keybindings
 * - History graph (bottom-left): species pop over time
 */
UCLASS()
class EVO4XSANDBOX_API ASandboxHUD : public AHUD
{
	GENERATED_BODY()

public:
	/** Set world state reference for rendering. */
	void SetWorldState(const FWorldState* InWorld) { CachedWorld = InWorld; }
	void SetSelectedRegion(int32 RegionId) { SelectedRegionId = RegionId; }

protected:
	virtual void DrawHUD() override;

private:
	const FWorldState* CachedWorld = nullptr;
	int32 SelectedRegionId = -1;

	// ── Drawing helpers ──

	void DrawGlobalPanel(float X, float Y);
	void DrawRegionInspector(float X, float Y);
	void DrawControlsHint(float X, float Y);
	void DrawHistoryGraph(float X, float Y, float W, float H);

	/** Draw text with shadow for readability. */
	void DrawTextShadow(const FString& Text, float X, float Y,
		FLinearColor Color = FLinearColor::White, float Scale = 1.f);

	/** Current speed as string. */
	FString SpeedString() const;

	/** Trend arrow character. */
	static TCHAR TrendChar(ETrend T);
};
