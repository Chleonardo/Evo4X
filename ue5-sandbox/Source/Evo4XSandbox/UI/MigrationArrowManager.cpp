#include "MigrationArrowManager.h"
#include "SimCore/SpeciesData.h"
#include "DrawDebugHelpers.h"

AMigrationArrowManager::AMigrationArrowManager()
{
	PrimaryActorTick.bCanEverTick = true;
}

FVector2D AMigrationArrowManager::GetRegionCenter(const FHexGrid& Grid, const FRegion& Region)
{
	FVector2D Sum(0.f, 0.f);
	for (const int32 HexIdx : Region.HexIndices)
	{
		Sum += Grid.Tiles[HexIdx].WorldPos;
	}
	return Region.HexIndices.Num() > 0
		? Sum / static_cast<float>(Region.HexIndices.Num())
		: FVector2D::ZeroVector;
}

void AMigrationArrowManager::UpdateArrows(const FHexGrid& Grid, const FWorldState& World)
{
	CachedArrows.Empty();

	if (World.MigrationEdgesThisTick.Num() == 0)
	{
		bNeedsRedraw = true;
		return;
	}

	// Find max migration count for normalization
	float MaxCount = 0.f;
	for (const FMigrationEdge& Edge : World.MigrationEdgesThisTick)
	{
		MaxCount = FMath::Max(MaxCount, Edge.Count);
	}

	if (MaxCount <= 0.f)
	{
		bNeedsRedraw = true;
		return;
	}

	// Build arrow data
	for (const FMigrationEdge& Edge : World.MigrationEdgesThisTick)
	{
		if (!World.Regions.IsValidIndex(Edge.FromRegion) ||
			!World.Regions.IsValidIndex(Edge.ToRegion))
			continue;

		const FVector2D FromCenter = GetRegionCenter(Grid, World.Regions[Edge.FromRegion]);
		const FVector2D ToCenter = GetRegionCenter(Grid, World.Regions[Edge.ToRegion]);

		// Shorten arrow slightly so it doesn't overlap labels
		const FVector2D Dir = (ToCenter - FromCenter).GetSafeNormal();
		const FVector2D AdjFrom = FromCenter + Dir * FHexGrid::HexSize * 0.5f;
		const FVector2D AdjTo = ToCenter - Dir * FHexGrid::HexSize * 0.5f;

		// Color by species
		FColor ArrowColor = FColor::White;
		const FSpeciesData& Spec = FSpeciesRegistry::Get(Edge.SpeciesId);
		ArrowColor = Spec.Color.ToFColor(true);

		// Thickness by count
		const float NormCount = Edge.Count / MaxCount;
		const float Thickness = FMath::Lerp(MinArrowThickness, MaxArrowThickness, NormCount);

		FArrowData Arrow;
		Arrow.Start = FVector(AdjFrom.X, AdjFrom.Y, ArrowZHeight);
		Arrow.End = FVector(AdjTo.X, AdjTo.Y, ArrowZHeight);
		Arrow.Color = ArrowColor;
		Arrow.Thickness = Thickness;
		CachedArrows.Add(Arrow);
	}

	bNeedsRedraw = true;
}

void AMigrationArrowManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// Draw arrows using debug lines (persistent for one frame)
	UWorld* World = GetWorld();
	if (!World) return;

	for (const FArrowData& Arrow : CachedArrows)
	{
		DrawDebugDirectionalArrow(
			World,
			Arrow.Start,
			Arrow.End,
			FHexGrid::HexSize * 0.3f, // arrowhead size
			Arrow.Color,
			false,   // not persistent
			-1.f,    // lifetime = this frame
			0,       // depth priority
			Arrow.Thickness
		);
	}
}
