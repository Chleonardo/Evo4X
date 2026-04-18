#include "HexMapRenderer.h"
#include "SimCore/BiomeData.h"
#include "SimCore/SpeciesData.h"
#include "MapGen/HexGrid.h"

AHexMapRenderer::AHexMapRenderer()
{
	PrimaryActorTick.bCanEverTick = false;

	MapMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("MapMesh"));
	RootComponent = MapMesh;
	MapMesh->bUseAsyncCooking = true;
}

void AHexMapRenderer::GenerateHexVertices(const FVector2D& Center, float Size,
	TArray<FVector>& OutVerts, TArray<int32>& OutTris, int32 BaseIndex)
{
	// Flat-top hexagon: 6 vertices + center = 7 vertices, 6 triangles
	// Angles for flat-top: 0, 60, 120, 180, 240, 300 degrees
	OutVerts.Add(FVector(Center.X, Center.Y, 0.f)); // center vertex

	for (int32 i = 0; i < 6; ++i)
	{
		const float AngleDeg = 60.f * i;
		const float AngleRad = FMath::DegreesToRadians(AngleDeg);
		const float X = Center.X + Size * FMath::Cos(AngleRad);
		const float Y = Center.Y + Size * FMath::Sin(AngleRad);
		OutVerts.Add(FVector(X, Y, 0.f));
	}

	// 6 triangles: center + vertex[i] + vertex[(i+1)%6]
	for (int32 i = 0; i < 6; ++i)
	{
		OutTris.Add(BaseIndex);                          // center
		OutTris.Add(BaseIndex + 1 + i);                  // current vertex
		OutTris.Add(BaseIndex + 1 + ((i + 1) % 6));      // next vertex
	}
}

void AHexMapRenderer::BuildMesh(const FHexGrid& Grid, const FWorldState& World)
{
	CachedGrid = &Grid;

	TArray<FVector> Vertices;
	TArray<int32> Triangles;
	TArray<FLinearColor> VertColors;
	TArray<FVector> Normals;
	TArray<FVector2D> UVs;

	const float HexSize = FHexGrid::HexSize * 0.95f; // slight gap between hexes

	for (const FHexTile& Tile : Grid.Tiles)
	{
		const int32 BaseIdx = Vertices.Num();
		GenerateHexVertices(Tile.WorldPos, HexSize, Vertices, Triangles, BaseIdx);

		// Color all 7 vertices of this hex the same
		const FLinearColor Color = GetHexColor(Tile, World);
		for (int32 i = 0; i < 7; ++i)
		{
			VertColors.Add(Color);
			Normals.Add(FVector(0.f, 0.f, 1.f));
			UVs.Add(FVector2D(0.f, 0.f));
		}
	}

	MapMesh->ClearAllMeshSections();
	MapMesh->CreateMeshSection_LinearColor(0, Vertices, Triangles, Normals, UVs, VertColors, TArray<FProcMeshTangent>(), false);
}

void AHexMapRenderer::UpdateColors(const FWorldState& World)
{
	if (!CachedGrid) return;

	TArray<FLinearColor> VertColors;
	VertColors.Reserve(CachedGrid->Tiles.Num() * 7);

	for (const FHexTile& Tile : CachedGrid->Tiles)
	{
		const FLinearColor Color = GetHexColor(Tile, World);
		for (int32 i = 0; i < 7; ++i)
		{
			VertColors.Add(Color);
		}
	}

	// Update only vertex colors (fast path)
	// We need to rebuild the section unfortunately, as UProceduralMeshComponent
	// doesn't have a "update colors only" method. But we can cache vertices.
	// For v1, just rebuild — 500 hexes is fast enough.
	if (MapMesh->GetNumSections() > 0)
	{
		FProcMeshSection* Section = MapMesh->GetProcMeshSection(0);
		if (Section && Section->ProcVertexBuffer.Num() == VertColors.Num())
		{
			for (int32 i = 0; i < VertColors.Num(); ++i)
			{
				Section->ProcVertexBuffer[i].Color = VertColors[i].ToFColor(true);
			}
			MapMesh->UpdateMeshSection(0, Section->ProcVertexBuffer, TArray<FVector>(), TArray<FVector>(), TArray<FVector2D>(), TArray<FColor>(), TArray<FProcMeshTangent>());
		}
	}
}

void AHexMapRenderer::SetOverlay(EMapOverlay Overlay, const FWorldState& World)
{
	CurrentOverlay = Overlay;
	UpdateColors(World);
}

FLinearColor AHexMapRenderer::GetHexColor(const FHexTile& Tile, const FWorldState& World) const
{
	if (!Tile.bIsLand)
	{
		return WaterColor();
	}

	const int32 RegionId = Tile.RegionId;
	if (RegionId < 0 || !World.Regions.IsValidIndex(RegionId))
	{
		return FLinearColor(0.3f, 0.3f, 0.3f); // unassigned
	}

	const FRegion& Region = World.Regions[RegionId];

	switch (CurrentOverlay)
	{
	case EMapOverlay::Biomes:
	{
		const FBiomeData& Biome = FBiomeRegistry::Get(Region.BiomeId);
		return Biome.Color;
	}

	case EMapOverlay::DominantSpecies:
	{
		if (Region.DominantSpecies.IsNone())
		{
			return FLinearColor(0.2f, 0.2f, 0.2f); // no species
		}
		const FSpeciesData& Spec = FSpeciesRegistry::Get(Region.DominantSpecies);
		return Spec.Color;
	}

	case EMapOverlay::TotalBiomass:
	{
		// Intensity based on biomass relative to region capacity
		const float MaxExpected = Region.HexCount * 20.f; // rough cap
		const float Intensity = FMath::Clamp(Region.TotalBiomass / MaxExpected, 0.f, 1.f);
		return FLinearColor(0.1f + Intensity * 0.5f, 0.3f + Intensity * 0.5f, 0.1f);
	}

	case EMapOverlay::ResourceR1:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(Region.BaseR1 / MaxRes, 0.f, 1.f) : 0.f;
		return FMath::Lerp(FLinearColor(0.2f, 0.2f, 0.2f), FLinearColor(0.5f, 0.9f, 0.3f), Intensity);
	}

	case EMapOverlay::ResourceR2:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(Region.BaseR2 / MaxRes, 0.f, 1.f) : 0.f;
		return FMath::Lerp(FLinearColor(0.2f, 0.2f, 0.2f), FLinearColor(0.1f, 0.5f, 0.1f), Intensity);
	}

	case EMapOverlay::ResourceR3:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(Region.BaseR3 / MaxRes, 0.f, 1.f) : 0.f;
		return FMath::Lerp(FLinearColor(0.2f, 0.2f, 0.2f), FLinearColor(0.7f, 0.6f, 0.2f), Intensity);
	}

	case EMapOverlay::ResourceTotal:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float TotalRes = Region.BaseR1 + Region.BaseR2 + Region.BaseR3;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(TotalRes / MaxRes, 0.f, 1.f) : 0.f;
		return FMath::Lerp(FLinearColor(0.15f, 0.15f, 0.15f), FLinearColor(0.8f, 0.8f, 0.4f), Intensity);
	}

	default:
		return FLinearColor(0.5f, 0.5f, 0.5f);
	}
}

int32 AHexMapRenderer::WorldPosToRegion(FVector WorldPos) const
{
	if (!CachedGrid) return -1;

	// Find closest land hex to the world position
	float BestDistSq = FLT_MAX;
	int32 BestRegion = -1;

	const FVector2D Pos2D(WorldPos.X, WorldPos.Y);

	for (const int32 LandIdx : CachedGrid->LandIndices)
	{
		const FHexTile& Tile = CachedGrid->Tiles[LandIdx];
		const float DistSq = FVector2D::DistSquared(Pos2D, Tile.WorldPos);

		if (DistSq < BestDistSq)
		{
			BestDistSq = DistSq;
			BestRegion = Tile.RegionId;
		}
	}

	// Only return if within hex radius
	if (BestDistSq < FHexGrid::HexSize * FHexGrid::HexSize * 1.5f)
	{
		return BestRegion;
	}

	return -1;
}
