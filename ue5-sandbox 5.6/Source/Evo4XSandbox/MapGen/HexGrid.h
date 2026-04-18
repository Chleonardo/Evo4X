#pragma once

#include "CoreMinimal.h"

/**
 * Hex coordinate using axial (q, r) system with flat-top hexagons.
 * Neighbors in 6 directions.
 */
struct FHexCoord
{
	int32 Q = 0;
	int32 R = 0;

	FHexCoord() = default;
	FHexCoord(int32 InQ, int32 InR) : Q(InQ), R(InR) {}

	bool operator==(const FHexCoord& Other) const { return Q == Other.Q && R == Other.R; }
	bool operator!=(const FHexCoord& Other) const { return !(*this == Other); }

	friend uint32 GetTypeHash(const FHexCoord& C)
	{
		return HashCombine(::GetTypeHash(C.Q), ::GetTypeHash(C.R));
	}

	/** S coordinate (cube system: q + r + s = 0). */
	int32 S() const { return -Q - R; }

	/** Distance between two hex coords (cube distance). */
	static int32 Distance(const FHexCoord& A, const FHexCoord& B)
	{
		return (FMath::Abs(A.Q - B.Q) + FMath::Abs(A.R - B.R) + FMath::Abs(A.S() - B.S())) / 2;
	}

	/** Get the 6 neighbors of this hex. */
	TArray<FHexCoord> Neighbors() const
	{
		return {
			FHexCoord(Q + 1, R),
			FHexCoord(Q - 1, R),
			FHexCoord(Q, R + 1),
			FHexCoord(Q, R - 1),
			FHexCoord(Q + 1, R - 1),
			FHexCoord(Q - 1, R + 1)
		};
	}
};

/**
 * A single hex tile in the world grid.
 */
struct FHexTile
{
	int32 Index = -1;          // index in the grid's Tiles array
	FHexCoord Coord;
	bool bIsLand = false;
	int32 RegionId = -1;       // -1 = unassigned / water

	/** World-space position (flat-top hex, z=0). */
	FVector2D WorldPos;
};

/**
 * Hex grid with continent generation.
 * Generates ~N land hexes forming a single continent surrounded by water.
 */
class EVO4XSANDBOX_API FHexGrid
{
public:
	/** Size of each hex (outer radius, center to vertex). */
	static constexpr float HexSize = 50.f;

	/** All tiles in the grid (land + water border). */
	TArray<FHexTile> Tiles;

	/** Fast lookup: HexCoord -> tile index. */
	TMap<FHexCoord, int32> CoordToIndex;

	/** Indices of land tiles only. */
	TArray<int32> LandIndices;

	/**
	 * Generate a continent of approximately TargetLandCount hexes.
	 * Uses noise-modulated flood fill from center for organic shape.
	 */
	void GenerateContinent(int32 TargetLandCount, const FString& Seed);

	/** Convert axial hex coord to world-space 2D position (flat-top). */
	static FVector2D HexToWorld(const FHexCoord& Coord);

	/** Get neighbor tile indices for a given tile (only existing tiles). */
	TArray<int32> GetTileNeighbors(int32 TileIndex) const;

	/** Get or create a tile at the given coordinate. Returns tile index. */
	int32 GetOrCreateTile(const FHexCoord& Coord);
};
