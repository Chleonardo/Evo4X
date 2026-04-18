#pragma once

#include "CoreMinimal.h"
#include "HexGrid.h"
#include "SimCore/WorldState.h"
#include "SimCore/SimConfig.h"

/**
 * Generates the complete world: hex grid, regions, biome assignment,
 * species seeding. Fully deterministic by seed.
 */
class EVO4XSANDBOX_API FWorldGenerator
{
public:
	/**
	 * Generate a complete world state ready for simulation.
	 * @param Config  Simulation configuration (seed, region count, etc.)
	 * @param OutGrid Populated hex grid (for rendering)
	 * @param OutWorld Populated world state (for simulation)
	 */
	static void Generate(const FSimConfig& Config, FHexGrid& OutGrid, FWorldState& OutWorld);

private:
	/** Assign land hexes to regions using weighted flood fill from seed points. */
	static void GenerateRegions(FHexGrid& Grid, int32 NumRegions, int32 MinSize, const FString& Seed);

	/** Merge any undersized regions into neighbors. */
	static void MergeSmallRegions(FHexGrid& Grid, int32 MinSize);

	/** Build adjacency graph between regions. */
	static void BuildAdjacency(const FHexGrid& Grid, TArray<FRegion>& Regions);

	/** Assign biomes to regions deterministically. */
	static void AssignBiomes(TArray<FRegion>& Regions, const FString& Seed);

	/** Seed initial species populations from biome packs. */
	static void SeedSpecies(TArray<FRegion>& Regions, const FSimConfig& Config);
};
