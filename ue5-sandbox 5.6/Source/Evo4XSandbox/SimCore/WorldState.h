#pragma once

#include "CoreMinimal.h"
#include "Region.h"
#include "SimConfig.h"

/** Simulation speed modes. */
enum class ESimSpeed : uint8
{
	Paused,
	Slow,      // 1 tick per TickSpeedSlow seconds
	Fast,      // 1 tick per TickSpeedFast seconds
	VeryFast   // 2 ticks per second
};

/** Global metrics snapshot (updated each tick). */
struct FGlobalMetrics
{
	float TotalWorldBiomass = 0.f;
	int32 AliveSpeciesCount = 0;
	FName FastestGrowing;
	FName FastestDeclining;
	TMap<FName, float> SpeciesTotalPop; // species -> world total pop
};

/**
 * Complete world state for the sandbox simulation.
 */
struct FWorldState
{
	int32 CurrentTick = 0;
	ESimSpeed Speed = ESimSpeed::Paused;

	FSimConfig Config;
	TArray<FRegion> Regions;

	// Migration buffer: populated during phase 1, applied in phase 2
	// RegionId -> (SpeciesId -> incoming count)
	TMap<int32, TMap<FName, float>> PendingMigrants;

	// All migration edges this tick (for arrow rendering)
	TArray<FMigrationEdge> MigrationEdgesThisTick;

	// Global metrics
	FGlobalMetrics Metrics;

	// Global population history per species (last N ticks)
	TMap<FName, TArray<float>> GlobalPopHistory;
};
