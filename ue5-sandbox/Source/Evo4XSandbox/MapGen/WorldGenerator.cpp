#include "WorldGenerator.h"
#include "SimCore/EvoRng.h"
#include "SimCore/BiomeData.h"
#include "SimCore/SpeciesData.h"
#include "Evo4XSandbox.h"

// ============================================================================
// Generate — full world creation pipeline
// ============================================================================

void FWorldGenerator::Generate(const FSimConfig& Config, FHexGrid& OutGrid, FWorldState& OutWorld)
{
	// Initialize registries
	FBiomeRegistry::Initialize();
	FSpeciesRegistry::Initialize();

	const FString& Seed = Config.WorldSeed;

	// Step 1: Generate hex continent
	OutGrid.GenerateContinent(Config.TotalHexes, Seed);
	UE_LOG(LogEvo4X, Log, TEXT("Generated continent: %d land hexes, %d total tiles"),
		OutGrid.LandIndices.Num(), OutGrid.Tiles.Num());

	// Step 2: Generate regions (assigns RegionId to each land hex)
	GenerateRegions(OutGrid, Config.NumRegions, Config.MinRegionSize, Seed);

	// Step 3: Merge small regions
	MergeSmallRegions(OutGrid, Config.MinRegionSize);

	// Count actual regions after merge
	TSet<int32> UniqueRegions;
	for (const int32 LandIdx : OutGrid.LandIndices)
	{
		if (OutGrid.Tiles[LandIdx].RegionId >= 0)
		{
			UniqueRegions.Add(OutGrid.Tiles[LandIdx].RegionId);
		}
	}

	// Step 4: Build Region structs
	// Remap region IDs to contiguous 0..N-1
	TMap<int32, int32> RegionRemap;
	int32 NextId = 0;
	for (const int32 OldId : UniqueRegions)
	{
		RegionRemap.Add(OldId, NextId++);
	}

	// Update hex tiles with remapped IDs
	for (const int32 LandIdx : OutGrid.LandIndices)
	{
		FHexTile& Tile = OutGrid.Tiles[LandIdx];
		if (const int32* NewId = RegionRemap.Find(Tile.RegionId))
		{
			Tile.RegionId = *NewId;
		}
	}

	const int32 NumRegions = NextId;
	OutWorld.Regions.SetNum(NumRegions);

	for (int32 i = 0; i < NumRegions; ++i)
	{
		OutWorld.Regions[i].RegionId = i;
		OutWorld.Regions[i].HexCount = 0;
	}

	// Populate hex indices per region
	for (const int32 LandIdx : OutGrid.LandIndices)
	{
		const int32 RId = OutGrid.Tiles[LandIdx].RegionId;
		if (RId >= 0 && RId < NumRegions)
		{
			OutWorld.Regions[RId].HexIndices.Add(LandIdx);
			OutWorld.Regions[RId].HexCount++;
		}
	}

	UE_LOG(LogEvo4X, Log, TEXT("Created %d regions"), NumRegions);

	// Step 5: Build adjacency graph
	BuildAdjacency(OutGrid, OutWorld.Regions);

	// Step 6: Assign biomes
	AssignBiomes(OutWorld.Regions, Seed);

	// Step 7: Compute base resources
	for (FRegion& Region : OutWorld.Regions)
	{
		Region.ComputeBaseResources(Config.RichnessPerHex);
	}

	// Step 8: Seed species
	SeedSpecies(OutWorld.Regions, Config);

	// Initialize world state
	OutWorld.Config = Config;
	OutWorld.CurrentTick = 0;
	OutWorld.Speed = ESimSpeed::Paused;

	UE_LOG(LogEvo4X, Log, TEXT("World generation complete. Seed: %s"), *Seed);
}

// ============================================================================
// GenerateRegions — weighted flood fill from seed points
// ============================================================================

void FWorldGenerator::GenerateRegions(FHexGrid& Grid, int32 NumRegions, int32 MinSize, const FString& Seed)
{
	if (Grid.LandIndices.Num() == 0) return;

	// Pick seed points: spread across land hexes
	// Use a simple approach: shuffle land indices, pick first NumRegions
	TArray<int32> ShuffledLand = FEvoRng::ShuffleIndices(Seed, TEXT("region_seeds"), Grid.LandIndices.Num());

	// We need to pick seed points that are well-spread.
	// Strategy: pick NumRegions points, ensuring minimum distance between them.
	TArray<int32> SeedTileIndices;
	const int32 MinSeedDist = FMath::Max(2, FMath::FloorToInt32(
		FMath::Sqrt(static_cast<float>(Grid.LandIndices.Num()) / NumRegions) * 0.5f));

	for (const int32 ShufIdx : ShuffledLand)
	{
		if (SeedTileIndices.Num() >= NumRegions) break;

		const int32 TileIdx = Grid.LandIndices[ShufIdx];
		const FHexCoord& Coord = Grid.Tiles[TileIdx].Coord;

		// Check distance to existing seeds
		bool bTooClose = false;
		for (const int32 ExistingSeedTile : SeedTileIndices)
		{
			const int32 Dist = FHexCoord::Distance(Coord, Grid.Tiles[ExistingSeedTile].Coord);
			if (Dist < MinSeedDist)
			{
				bTooClose = true;
				break;
			}
		}

		if (!bTooClose)
		{
			SeedTileIndices.Add(TileIdx);
		}
	}

	// If we couldn't find enough well-spaced seeds, fill remaining from shuffled list
	if (SeedTileIndices.Num() < NumRegions)
	{
		for (const int32 ShufIdx : ShuffledLand)
		{
			if (SeedTileIndices.Num() >= NumRegions) break;
			const int32 TileIdx = Grid.LandIndices[ShufIdx];
			if (!SeedTileIndices.Contains(TileIdx))
			{
				SeedTileIndices.Add(TileIdx);
			}
		}
	}

	const int32 ActualRegions = SeedTileIndices.Num();

	// Assign seed tiles
	for (int32 i = 0; i < ActualRegions; ++i)
	{
		Grid.Tiles[SeedTileIndices[i]].RegionId = i;
	}

	// Flood fill expansion: each region grows from its seed
	// Use a multi-source BFS where each region expands in turns
	TArray<TArray<int32>> RegionFrontiers;
	RegionFrontiers.SetNum(ActualRegions);
	for (int32 i = 0; i < ActualRegions; ++i)
	{
		RegionFrontiers[i].Add(SeedTileIndices[i]);
	}

	bool bChanged = true;
	int32 Round = 0;

	while (bChanged)
	{
		bChanged = false;

		// Each region tries to claim one neighbor per round (roughly uniform growth)
		for (int32 RId = 0; RId < ActualRegions; ++RId)
		{
			if (RegionFrontiers[RId].Num() == 0) continue;

			// Collect all unassigned neighbors of frontier
			TArray<int32> Candidates;
			for (const int32 FrontTile : RegionFrontiers[RId])
			{
				TArray<int32> Nbs = Grid.GetTileNeighbors(FrontTile);
				for (const int32 NbIdx : Nbs)
				{
					if (Grid.Tiles[NbIdx].bIsLand && Grid.Tiles[NbIdx].RegionId < 0)
					{
						Candidates.AddUnique(NbIdx);
					}
				}
			}

			if (Candidates.Num() == 0)
			{
				RegionFrontiers[RId].Empty();
				continue;
			}

			// Pick one candidate deterministically
			const FString PickKey = FString::Printf(TEXT("region_expand|%d|%d"), RId, Round);
			const int32 PickIdx = FEvoRng::ChooseIndex(Seed, PickKey, Candidates.Num());
			const int32 ClaimedTile = Candidates[PickIdx];

			Grid.Tiles[ClaimedTile].RegionId = RId;
			RegionFrontiers[RId].Add(ClaimedTile);
			bChanged = true;
		}

		Round++;

		// Safety
		if (Round > Grid.LandIndices.Num() * 2)
		{
			break;
		}
	}

	// Assign any remaining unassigned land tiles to nearest region
	for (const int32 LandIdx : Grid.LandIndices)
	{
		if (Grid.Tiles[LandIdx].RegionId >= 0) continue;

		// Find nearest assigned neighbor
		int32 BestRegion = 0;
		int32 BestDist = INT_MAX;
		for (int32 i = 0; i < ActualRegions; ++i)
		{
			const int32 Dist = FHexCoord::Distance(Grid.Tiles[LandIdx].Coord,
				Grid.Tiles[SeedTileIndices[i]].Coord);
			if (Dist < BestDist)
			{
				BestDist = Dist;
				BestRegion = i;
			}
		}
		Grid.Tiles[LandIdx].RegionId = BestRegion;
	}
}

// ============================================================================
// MergeSmallRegions
// ============================================================================

void FWorldGenerator::MergeSmallRegions(FHexGrid& Grid, int32 MinSize)
{
	// Count hexes per region
	TMap<int32, int32> RegionSizes;
	for (const int32 LandIdx : Grid.LandIndices)
	{
		const int32 RId = Grid.Tiles[LandIdx].RegionId;
		RegionSizes.FindOrAdd(RId)++;
	}

	// Find small regions and merge them into their largest neighbor
	bool bMerged = true;
	while (bMerged)
	{
		bMerged = false;
		RegionSizes.Empty();
		for (const int32 LandIdx : Grid.LandIndices)
		{
			RegionSizes.FindOrAdd(Grid.Tiles[LandIdx].RegionId)++;
		}

		for (const auto& Pair : RegionSizes)
		{
			if (Pair.Value >= MinSize) continue;

			const int32 SmallRegion = Pair.Key;

			// Find neighboring regions
			TMap<int32, int32> NeighborRegionSizes;
			for (const int32 LandIdx : Grid.LandIndices)
			{
				if (Grid.Tiles[LandIdx].RegionId != SmallRegion) continue;

				TArray<int32> Nbs = Grid.GetTileNeighbors(LandIdx);
				for (const int32 NbIdx : Nbs)
				{
					if (Grid.Tiles[NbIdx].bIsLand && Grid.Tiles[NbIdx].RegionId != SmallRegion)
					{
						const int32 NbRegion = Grid.Tiles[NbIdx].RegionId;
						NeighborRegionSizes.FindOrAdd(NbRegion) = RegionSizes.FindRef(NbRegion);
					}
				}
			}

			if (NeighborRegionSizes.Num() == 0) continue;

			// Merge into largest neighbor
			int32 BestNeighbor = -1;
			int32 BestSize = -1;
			for (const auto& NbPair : NeighborRegionSizes)
			{
				if (NbPair.Value > BestSize)
				{
					BestSize = NbPair.Value;
					BestNeighbor = NbPair.Key;
				}
			}

			// Reassign all hexes from small region to best neighbor
			for (const int32 LandIdx : Grid.LandIndices)
			{
				if (Grid.Tiles[LandIdx].RegionId == SmallRegion)
				{
					Grid.Tiles[LandIdx].RegionId = BestNeighbor;
				}
			}

			bMerged = true;
			break; // Restart loop after merge
		}
	}
}

// ============================================================================
// BuildAdjacency
// ============================================================================

void FWorldGenerator::BuildAdjacency(const FHexGrid& Grid, TArray<FRegion>& Regions)
{
	for (FRegion& Region : Regions)
	{
		Region.Neighbors.Empty();
	}

	// For each land hex, check if any neighbor belongs to a different region
	TSet<uint64> SeenPairs; // packed pair to avoid duplicates

	for (const int32 LandIdx : Grid.LandIndices)
	{
		const int32 RId = Grid.Tiles[LandIdx].RegionId;
		if (RId < 0) continue;

		TArray<int32> Nbs = Grid.GetTileNeighbors(LandIdx);
		for (const int32 NbIdx : Nbs)
		{
			if (!Grid.Tiles[NbIdx].bIsLand) continue;
			const int32 NbRegion = Grid.Tiles[NbIdx].RegionId;
			if (NbRegion < 0 || NbRegion == RId) continue;

			// Pack pair as uint64 for dedup
			const int32 Lo = FMath::Min(RId, NbRegion);
			const int32 Hi = FMath::Max(RId, NbRegion);
			const uint64 PairKey = (static_cast<uint64>(Lo) << 32) | static_cast<uint64>(Hi);

			if (!SeenPairs.Contains(PairKey))
			{
				SeenPairs.Add(PairKey);
				if (Regions.IsValidIndex(RId))
				{
					Regions[RId].Neighbors.AddUnique(NbRegion);
				}
				if (Regions.IsValidIndex(NbRegion))
				{
					Regions[NbRegion].Neighbors.AddUnique(RId);
				}
			}
		}
	}

	for (FRegion& Region : Regions)
	{
		UE_LOG(LogEvo4X, Verbose, TEXT("Region %d: %d hexes, %d neighbors"),
			Region.RegionId, Region.HexCount, Region.Neighbors.Num());
	}
}

// ============================================================================
// AssignBiomes
// ============================================================================

void FWorldGenerator::AssignBiomes(TArray<FRegion>& Regions, const FString& Seed)
{
	const TArray<FName>& BiomeIds = FBiomeRegistry::GetAllIds();
	const int32 NumBiomes = BiomeIds.Num();

	for (int32 i = 0; i < Regions.Num(); ++i)
	{
		const FString Key = FString::Printf(TEXT("biome_assign|%d"), i);
		const int32 BiomeIdx = FEvoRng::ChooseIndex(Seed, Key, NumBiomes);
		Regions[i].BiomeId = BiomeIds[BiomeIdx];

		UE_LOG(LogEvo4X, Log, TEXT("Region %d -> biome: %s (%d hexes)"),
			i, *Regions[i].BiomeId.ToString(), Regions[i].HexCount);
	}
}

// ============================================================================
// SeedSpecies
// ============================================================================

void FWorldGenerator::SeedSpecies(TArray<FRegion>& Regions, const FSimConfig& Config)
{
	const FString& Seed = Config.WorldSeed;

	for (int32 i = 0; i < Regions.Num(); ++i)
	{
		FRegion& Region = Regions[i];
		const FBiomeData& Biome = FBiomeRegistry::Get(Region.BiomeId);

		if (Biome.Packs.Num() == 0) continue;

		// Pick a pack deterministically
		const FString PackKey = FString::Printf(TEXT("species_pack|%d"), i);

		// Filter out empty packs
		TArray<int32> NonEmptyPackIndices;
		for (int32 p = 0; p < Biome.Packs.Num(); ++p)
		{
			if (Biome.Packs[p].Num() > 0)
			{
				NonEmptyPackIndices.Add(p);
			}
		}

		if (NonEmptyPackIndices.Num() == 0) continue;

		const int32 PickIdx = FEvoRng::ChooseIndex(Seed, PackKey, NonEmptyPackIndices.Num());
		const TArray<FName>& Pack = Biome.Packs[NonEmptyPackIndices[PickIdx]];

		// Seed each species in the pack
		const float StartPop = Region.HexCount * Config.StartPopDensity;

		for (const FName& SpeciesId : Pack)
		{
			Region.Populations.Add(SpeciesId, StartPop);
		}

		UE_LOG(LogEvo4X, Log, TEXT("Region %d (%s): seeded %d species, pop=%.0f each"),
			i, *Region.BiomeId.ToString(), Pack.Num(), StartPop);
	}
}
