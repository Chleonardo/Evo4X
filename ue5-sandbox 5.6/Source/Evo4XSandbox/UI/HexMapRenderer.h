#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h"
#include "Materials/MaterialInterface.h"
#include "MapGen/HexGrid.h"
#include "SimCore/WorldState.h"
#include "HexMapRenderer.generated.h"

/** Overlay mode for the map. */
UENUM(BlueprintType)
enum class EMapOverlay : uint8
{
	Biomes,
	DominantSpecies,
	TotalBiomass,
	ResourceR1,
	ResourceR2,
	ResourceR3,
	ResourceTotal
};

/**
 * Renders the hex map as a flat procedural mesh.
 * Each hex is a separate section with its own vertex color.
 */
UCLASS()
class EVO4XSANDBOX_API AHexMapRenderer : public AActor
{
	GENERATED_BODY()

public:
	AHexMapRenderer();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	UProceduralMeshComponent* MapMesh;

	/** Assign VertexColorMaterial here in the editor Details panel. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Map")
	UMaterialInterface* HexMaterial = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Map")
	EMapOverlay CurrentOverlay = EMapOverlay::Biomes;

	/** Build the entire hex mesh from grid data. Call once after world gen. */
	void BuildMesh(const FHexGrid& Grid, const FWorldState& World);

	/** Update vertex colors based on current overlay mode and world state. */
	void UpdateColors(const FWorldState& World);

	/** Set overlay mode and refresh colors. */
	void SetOverlay(EMapOverlay Overlay, const FWorldState& World);

	/** Hit test: convert world position to region ID. Returns -1 if water/miss. */
	UFUNCTION(BlueprintCallable, Category = "Map")
	int32 WorldPosToRegion(FVector WorldPos) const;

protected:
	virtual void BeginPlay() override;

private:
	/** Cached grid reference for hit testing and color updates. */
	const FHexGrid* CachedGrid = nullptr;

	/** Cached vertex color material (loaded once in BeginPlay). */
	UPROPERTY()
	UMaterialInterface* CachedMaterial = nullptr;

	/** Generate hex vertices for a single hex at given center. */
	static void GenerateHexVertices(const FVector2D& Center, float Size,
		TArray<FVector>& OutVerts, TArray<int32>& OutTris, int32 BaseIndex);

	/** Get color for a hex based on current overlay. */
	FLinearColor GetHexColor(const FHexTile& Tile, const FWorldState& World) const;

	/** Water color. */
	static FLinearColor WaterColor() { return FLinearColor(0.15f, 0.30f, 0.55f); }
};
