#include "HexGrid.h"
#include "SimCore/EvoRng.h"

FVector2D FHexGrid::HexToWorld(const FHexCoord& Coord)
{
	// Flat-top hex layout
	const float X = HexSize * (3.f / 2.f * Coord.Q);
	const float Y = HexSize * (FMath::Sqrt(3.f) * (Coord.R + Coord.Q / 2.f));
	return FVector2D(X, Y);
}

int32 FHexGrid::GetOrCreateTile(const FHexCoord& Coord)
{
	if (const int32* Existing = CoordToIndex.Find(Coord))
	{
		return *Existing;
	}

	const int32 Idx = Tiles.Num();
	FHexTile Tile;
	Tile.Index = Idx;
	Tile.Coord = Coord;
	Tile.WorldPos = HexToWorld(Coord);
	Tiles.Add(MoveTemp(Tile));
	CoordToIndex.Add(Coord, Idx);
	return Idx;
}

TArray<int32> FHexGrid::GetTileNeighbors(int32 TileIndex) const
{
	TArray<int32> Result;
	if (!Tiles.IsValidIndex(TileIndex)) return Result;

	const FHexCoord& Coord = Tiles[TileIndex].Coord;
	TArray<FHexCoord> Nbs = Coord.Neighbors();

	for (const FHexCoord& Nb : Nbs)
	{
		if (const int32* Idx = CoordToIndex.Find(Nb))
		{
			Result.Add(*Idx);
		}
	}
	return Result;
}

void FHexGrid::GenerateContinent(int32 TargetLandCount, const FString& Seed)
{
	Tiles.Empty();
	CoordToIndex.Empty();
	LandIndices.Empty();

	// Start from center
	const FHexCoord Center(0, 0);
	const int32 CenterIdx = GetOrCreateTile(Center);
	Tiles[CenterIdx].bIsLand = true;
	LandIndices.Add(CenterIdx);

	// Frontier for flood fill expansion
	TArray<int32> Frontier;
	Frontier.Add(CenterIdx);

	int32 Iteration = 0;

	while (LandIndices.Num() < TargetLandCount && Frontier.Num() > 0)
	{
		// Pick a frontier tile to expand from
		// Weight toward tiles closer to center for organic shape
		const FString PickKey = FString::Printf(TEXT("continent|pick|%d"), Iteration);
		const int32 FrontierIdx = FEvoRng::ChooseIndex(Seed, PickKey, Frontier.Num());
		const int32 TileIdx = Frontier[FrontierIdx];
		const FHexCoord& TileCoord = Tiles[TileIdx].Coord;

		// Try to expand to a random neighbor
		TArray<FHexCoord> Nbs = TileCoord.Neighbors();
		const FString NbKey = FString::Printf(TEXT("continent|nb|%d"), Iteration);
		const int32 NbPick = FEvoRng::ChooseIndex(Seed, NbKey, Nbs.Num());
		const FHexCoord& NbCoord = Nbs[NbPick];

		// Distance check: don't grow too far from center (creates organic blob)
		const int32 Dist = FHexCoord::Distance(Center, NbCoord);
		const int32 MaxRadius = FMath::CeilToInt32(FMath::Sqrt(static_cast<float>(TargetLandCount)) * 1.2f);

		// Probability of accepting decreases with distance (noise-like shape)
		const FString AcceptKey = FString::Printf(TEXT("continent|accept|%d"), Iteration);
		const float AcceptRoll = FEvoRng::Rand01(Seed, AcceptKey);
		const float AcceptChance = FMath::Clamp(1.f - static_cast<float>(Dist) / MaxRadius, 0.1f, 1.f);

		Iteration++;

		if (Dist > MaxRadius + 2)
		{
			// Too far, remove from frontier if all neighbors are land or too far
			continue;
		}

		if (AcceptRoll > AcceptChance)
		{
			continue; // Rejected by noise
		}

		// Create or find the neighbor tile
		const int32 NbIdx = GetOrCreateTile(NbCoord);

		if (!Tiles[NbIdx].bIsLand)
		{
			Tiles[NbIdx].bIsLand = true;
			LandIndices.Add(NbIdx);
			Frontier.Add(NbIdx);
		}

		// Remove frontier tile if all its neighbors are land
		bool bAllNeighborsLand = true;
		TArray<FHexCoord> CheckNbs = TileCoord.Neighbors();
		for (const FHexCoord& CheckNb : CheckNbs)
		{
			const int32* CheckIdx = CoordToIndex.Find(CheckNb);
			if (!CheckIdx || !Tiles[*CheckIdx].bIsLand)
			{
				bAllNeighborsLand = false;
				break;
			}
		}
		if (bAllNeighborsLand)
		{
			Frontier.RemoveSwap(FrontierIdx);
		}

		// Safety: prevent infinite loop
		if (Iteration > TargetLandCount * 20)
		{
			break;
		}
	}

	// Create water border tiles (1 ring around land for rendering)
	TSet<FHexCoord> WaterCoords;
	for (const int32 LandIdx : LandIndices)
	{
		TArray<FHexCoord> Nbs = Tiles[LandIdx].Coord.Neighbors();
		for (const FHexCoord& Nb : Nbs)
		{
			if (!CoordToIndex.Contains(Nb))
			{
				WaterCoords.Add(Nb);
			}
			else if (!Tiles[CoordToIndex[Nb]].bIsLand)
			{
				// Already exists as water, fine
			}
		}
	}

	for (const FHexCoord& WCoord : WaterCoords)
	{
		GetOrCreateTile(WCoord); // bIsLand defaults to false
	}
}
