#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/TextRenderComponent.h"
#include "SimCore/WorldState.h"
#include "MapGen/HexGrid.h"
#include "RegionOverlay.generated.h"

/**
 * Renders per-region text overlays on the map:
 * - Dominant species abbreviation
 * - Trend arrow
 * - Population count
 * Spawns one TextRenderComponent per region.
 */
UCLASS()
class EVO4XSANDBOX_API ARegionOverlay : public AActor
{
	GENERATED_BODY()

public:
	ARegionOverlay();

	/** Initialize overlay labels for all regions. Call once after world gen. */
	void InitOverlays(const FHexGrid& Grid, const FWorldState& World);

	/** Update overlay text from current world state. Call each tick. */
	void UpdateOverlays(const FWorldState& World);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Overlay")
	float TextScale = 3.f;

private:
	UPROPERTY()
	TArray<UTextRenderComponent*> RegionLabels;

	/** Compute center position of a region (average of its hex positions). */
	static FVector2D ComputeRegionCenter(const FHexGrid& Grid, const FRegion& Region);

	/** Format overlay text for a region. */
	static FString FormatRegionText(const FRegion& Region);

	/** Get trend arrow string. */
	static FString TrendArrow(ETrend T);
};
