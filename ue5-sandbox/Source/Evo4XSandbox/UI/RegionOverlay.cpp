#include "RegionOverlay.h"
#include "SimCore/SpeciesData.h"
#include "MapGen/HexGrid.h"

ARegionOverlay::ARegionOverlay()
{
	PrimaryActorTick.bCanEverTick = false;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

FVector2D ARegionOverlay::ComputeRegionCenter(const FHexGrid& Grid, const FRegion& Region)
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

FString ARegionOverlay::TrendArrow(ETrend T)
{
	switch (T)
	{
	case ETrend::Up:   return TEXT("^");
	case ETrend::Down: return TEXT("v");
	default:           return TEXT("-");
	}
}

FString ARegionOverlay::FormatRegionText(const FRegion& Region)
{
	if (Region.DominantSpecies.IsNone() || Region.TotalBiomass <= 0.f)
	{
		return TEXT("--");
	}

	const FSpeciesData& Spec = FSpeciesRegistry::Get(Region.DominantSpecies);
	const FString Trend = TrendArrow(Region.BiomasssTrend);
	const int32 Pop = FMath::RoundToInt32(Region.TotalBiomass);

	return FString::Printf(TEXT("%s %s\n%d"), *Spec.DisplayChar, *Trend, Pop);
}

void ARegionOverlay::InitOverlays(const FHexGrid& Grid, const FWorldState& World)
{
	// Clear existing
	for (UTextRenderComponent* Label : RegionLabels)
	{
		if (Label) Label->DestroyComponent();
	}
	RegionLabels.Empty();

	for (const FRegion& Region : World.Regions)
	{
		const FVector2D Center = ComputeRegionCenter(Grid, Region);

		UTextRenderComponent* Label = NewObject<UTextRenderComponent>(this);
		Label->SetupAttachment(RootComponent);
		Label->RegisterComponent();
		Label->SetWorldLocation(FVector(Center.X, Center.Y, 5.f)); // slightly above hex plane
		Label->SetWorldRotation(FRotator(90.f, 0.f, 0.f)); // face up (top-down camera)
		Label->SetHorizontalAlignment(EHTA_Center);
		Label->SetVerticalAlignment(EVRTA_TextCenter);
		Label->SetWorldSize(FHexGrid::HexSize * TextScale);
		Label->SetTextRenderColor(FColor::White);
		Label->SetText(FText::FromString(FormatRegionText(Region)));

		RegionLabels.Add(Label);
	}
}

void ARegionOverlay::UpdateOverlays(const FWorldState& World)
{
	for (int32 i = 0; i < RegionLabels.Num() && i < World.Regions.Num(); ++i)
	{
		if (RegionLabels[i])
		{
			const FString Text = FormatRegionText(World.Regions[i]);
			RegionLabels[i]->SetText(FText::FromString(Text));

			// Tint text by dominant species color for readability
			if (!World.Regions[i].DominantSpecies.IsNone())
			{
				const FSpeciesData& Spec = FSpeciesRegistry::Get(World.Regions[i].DominantSpecies);
				RegionLabels[i]->SetTextRenderColor(Spec.Color.ToFColor(true));
			}
			else
			{
				RegionLabels[i]->SetTextRenderColor(FColor(128, 128, 128));
			}
		}
	}
}
