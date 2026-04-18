#pragma once

#include "CoreMinimal.h"
#include "WorldState.h"

/**
 * Pure simulation engine — no UE rendering dependencies.
 * Operates on FWorldState, fully deterministic by seed.
 *
 * Tick order (from spec):
 *  1. Apply active effects (if enabled)
 *  2. Births
 *  3. Resource allocation
 *  4. Identify starving newborns
 *  5. Compute outgoing expeditions -> PendingMigrants
 *  6. Local starvation commit
 *  7. Apply incoming migrants
 *  8. Extinction cleanup
 *  9. Metrics / trends / history
 * 10. Advance tick
 */
class EVO4XSANDBOX_API FSimulationCore
{
public:
	/** Run one full simulation tick on the world state. */
	static void SimulateTick(FWorldState& World);

private:
	// ── Phase 1: per-region, all regions before phase 2 ──

	/** Step 1: Apply active effects to effective resources. */
	static void ApplyActiveEffects(FRegion& Region);

	/** Step 2: Compute births for all species in a region. */
	static void ComputeBirths(FRegion& Region);

	/** Step 3: Allocate resources to species proportionally. */
	static void AllocateResources(FRegion& Region);

	/** Steps 4-5: Identify starving newborns, compute outgoing expeditions. */
	static void ComputeMigration(FWorldState& World, FRegion& Region);

	/** Step 6: Commit local starvation — survivors = min(n_pre, food). */
	static void CommitStarvation(FRegion& Region, const FSimConfig& Config);

	// ── Phase 2: global ──

	/** Step 7: Apply all pending incoming migrants. */
	static void ApplyIncomingMigrants(FWorldState& World);

	/** Step 8: Remove extinct species from regions. */
	static void ExtinctionCleanup(FWorldState& World);

	/** Step 9: Update metrics, trends, history. */
	static void UpdateMetrics(FWorldState& World);

	/** Tick down effect TTLs, remove expired. */
	static void TickDownEffects(FRegion& Region);
};
