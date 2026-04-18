#include "HexMapRenderer.h"
#include "SimCore/BiomeData.h"
#include "SimCore/SpeciesData.h"
#include "MapGen/HexGrid.h"
#include "Evo4XSandbox.h"
#include "Materials/MaterialInterface.h"
#if WITH_EDITOR
#include "Materials/Material.h"
#include "Materials/MaterialExpressionVertexColor.h"
#endif

AHexMapRenderer::AHexMapRenderer()
{
	PrimaryActorTick.bCanEverTick = false;

	MapMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("MapMesh"));
	RootComponent = MapMesh;
	MapMesh->bUseAsyncCooking = true;
}

void AHexMapRenderer::BeginPlay()
{
	Super::BeginPlay();

	// Resolve and cache the material once
	if (HexMaterial)
	{
		CachedMaterial = HexMaterial;
	}
	else
	{
		static const TCHAR* Candidates[] = {
			TEXT("/Game/M_HexMap"),
			TEXT("/Engine/EngineDebugMaterials/VertexColorMaterial"),
			TEXT("/Engine/EngineMaterials/VertexColorMaterial"),
		};
		for (const TCHAR* Path : Candidates)
		{
			if (UMaterialInterface* Mat = LoadObject<UMaterialInterface>(nullptr, Path))
			{
				CachedMaterial = Mat;
				break;
			}
		}
	}

	if (CachedMaterial)
	{
		MapMesh->SetMaterial(0, CachedMaterial);
		return;
	}

#if WITH_EDITOR
	// Fallback: create a minimal unlit vertex-color material at runtime (editor only)
	UMaterial* DynMat = NewObject<UMaterial>(GetTransientPackage(), NAME_None, RF_Transient);

	UMaterialExpressionVertexColor* VCExpr = NewObject<UMaterialExpressionVertexColor>(DynMat);
	auto* EditorData = DynMat->GetEditorOnlyData();
	EditorData->ExpressionCollection.AddExpression(VCExpr);

	// Drive EmissiveColor so it renders unlit
	EditorData->EmissiveColor.Expression = VCExpr;
	EditorData->EmissiveColor.OutputIndex = 0;

	DynMat->SetShadingModel(MSM_Unlit);

	DynMat->PostEditChange();
	MapMesh->SetMaterial(0, DynMat);
#endif
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

	// 6 triangles: CCW from below = front face pointing up (+Z), visible from top-down camera
	for (int32 i = 0; i < 6; ++i)
	{
		OutTris.Add(BaseIndex);                          // center
		OutTris.Add(BaseIndex + 1 + ((i + 1) % 6));      // next vertex (swapped for CCW)
		OutTris.Add(BaseIndex + 1 + i);                  // current vertex
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

	if (CachedMaterial)
	{
		MapMesh->SetMaterial(0, CachedMaterial);
	}
}

void AHexMapRenderer::UpdateColors(const FWorldState& World)
{
	if (!CachedGrid) return;
	// Full rebuild guarantees identical rendering to BuildMesh
	BuildMesh(*CachedGrid, World);
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

	// Border detection: 0 = interior, 1 = inner border (touches different region), 2 = outer border (1 step inside)
	int32 BorderRing = 0;
	if (CachedGrid)
	{
		for (const int32 NbIdx : CachedGrid->GetTileNeighbors(Tile.Index))
		{
			const FHexTile& Nb = CachedGrid->Tiles[NbIdx];
			if (!Nb.bIsLand || Nb.RegionId != RegionId)
			{
				BorderRing = 1; // this tile directly touches a border
				break;
			}
		}

		if (BorderRing == 0)
		{
			// Check if any neighbor is itself a border tile (ring 2)
			for (const int32 NbIdx : CachedGrid->GetTileNeighbors(Tile.Index))
			{
				if (!CachedGrid->Tiles[NbIdx].bIsLand) continue;
				for (const int32 Nb2Idx : CachedGrid->GetTileNeighbors(NbIdx))
				{
					const FHexTile& Nb2 = CachedGrid->Tiles[Nb2Idx];
					if (!Nb2.bIsLand || Nb2.RegionId != RegionId)
					{
						BorderRing = 2;
						break;
					}
				}
				if (BorderRing == 2) break;
			}
		}
	}

	const FRegion& Region = World.Regions[RegionId];

	FLinearColor Color(0.5f, 0.5f, 0.5f);

	switch (CurrentOverlay)
	{
	case EMapOverlay::Biomes:
	{
		const FBiomeData& Biome = FBiomeRegistry::Get(Region.BiomeId);
		Color = Biome.Color;
		break;
	}
	case EMapOverlay::DominantSpecies:
	{
		if (Region.DominantSpecies.IsNone())
		{
			Color = FLinearColor(0.2f, 0.2f, 0.2f);
		}
		else
		{
			const FSpeciesData& Spec = FSpeciesRegistry::Get(Region.DominantSpecies);
			Color = Spec.Color;
		}
		break;
	}
	case EMapOverlay::TotalBiomass:
	{
		const float MaxExpected = Region.HexCount * 20.f;
		const float Intensity = FMath::Clamp(Region.TotalBiomass / MaxExpected, 0.f, 1.f);
		Color = FLinearColor(0.1f + Intensity * 0.5f, 0.3f + Intensity * 0.5f, 0.1f);
		break;
	}
	case EMapOverlay::ResourceR1:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(Region.BaseR1 / MaxRes, 0.f, 1.f) : 0.f;
		Color = FMath::Lerp(FLinearColor(0.2f, 0.2f, 0.2f), FLinearColor(0.5f, 0.9f, 0.3f), Intensity);
		break;
	}
	case EMapOverlay::ResourceR2:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(Region.BaseR2 / MaxRes, 0.f, 1.f) : 0.f;
		Color = FMath::Lerp(FLinearColor(0.2f, 0.2f, 0.2f), FLinearColor(0.1f, 0.5f, 0.1f), Intensity);
		break;
	}
	case EMapOverlay::ResourceR3:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(Region.BaseR3 / MaxRes, 0.f, 1.f) : 0.f;
		Color = FMath::Lerp(FLinearColor(0.2f, 0.2f, 0.2f), FLinearColor(0.7f, 0.6f, 0.2f), Intensity);
		break;
	}
	case EMapOverlay::ResourceTotal:
	{
		const float MaxRes = Region.HexCount * World.Config.RichnessPerHex;
		const float TotalRes = Region.BaseR1 + Region.BaseR2 + Region.BaseR3;
		const float Intensity = MaxRes > 0.f ? FMath::Clamp(TotalRes / MaxRes, 0.f, 1.f) : 0.f;
		Color = FMath::Lerp(FLinearColor(0.15f, 0.15f, 0.15f), FLinearColor(0.8f, 0.8f, 0.4f), Intensity);
		break;
	}
	default: break;
	}

	// Darken border hexes for region outline effect
	if (BorderRing == 1)      Color *= 0.30f; // inner border — very dark
	else if (BorderRing == 2) Color *= 0.65f; // outer border — slightly dark

	return Color;
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
