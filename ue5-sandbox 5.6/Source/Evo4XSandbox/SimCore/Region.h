#pragma once

#include "CoreMinimal.h"

/** Trend direction for UI display. */
enum class ETrend : uint8
{
	Flat,
	Up,
	Down
};

/** Active effect on a region (architecture-ready, unused in v1). */
struct FActiveEffect
{
	FName EffectType;     // e.g. "fluctuation"
	float MultR1 = 1.f;
	float MultR2 = 1.f;
	float MultR3 = 1.f;
	int32 TTL = 0;        // ticks remaining
};

/** Per-species snapshot within a region for the current tick. */
struct FSpeciesInRegion
{
	float Population = 0.f;
	float PopPreBirth = 0.f;    // n_pre = pop * (1 + r), set during tick
	float FoodAllocated = 0.f;  // food received from allocation
	float StarvingTotal = 0.f;
	float StarvingNewborns = 0.f;
	float OutgoingMigrants = 0.f;
	float IncomingMigrants = 0.f;
	ETrend Trend = ETrend::Flat;
};

/** Migration record for one species going from one region to a neighbor. */
struct FMigrationEdge
{
	int32 FromRegion = -1;
	int32 ToRegion = -1;
	FName SpeciesId;
	float Count = 0.f;   // number of migrants
};

/**
 * A simulation region — the fundamental spatial unit.
 * Composed of multiple hex tiles, but simulated as one.
 */
struct FRegion
{
	int32 RegionId = -1;
	FName BiomeId;
	int32 HexCount = 0;

	// Base resources (computed from biome + hex count + richness)
	float BaseR1 = 0.f;
	float BaseR2 = 0.f;
	float BaseR3 = 0.f;

	// Effective resources this tick (after effects)
	float EffR1 = 0.f;
	float EffR2 = 0.f;
	float EffR3 = 0.f;

	// Neighbor region IDs (adjacency graph)
	TArray<int32> Neighbors;

	// Species populations: SpeciesId -> current population
	TMap<FName, float> Populations;

	// Per-tick breakdown (populated during SimulateTick)
	TMap<FName, FSpeciesInRegion> TickBreakdown;

	// Active effects (empty in v1)
	TArray<FActiveEffect> ActiveEffects;

	// Population history per species (ring buffer for trends)
	TMap<FName, TArray<float>> PopHistory;

	// Hex indices belonging to this region (for rendering)
	TArray<int32> HexIndices;

	// Cached: dominant species this tick
	FName DominantSpecies;
	float TotalBiomass = 0.f;
	ETrend BiomasssTrend = ETrend::Flat;

	/** Compute base resources from biome data and hex count. */
	void ComputeBaseResources(float RichnessPerHex);

	/** Get total population across all species. */
	float GetTotalPopulation() const;

	/** Find the species with the highest population. */
	FName FindDominantSpecies() const;
};
