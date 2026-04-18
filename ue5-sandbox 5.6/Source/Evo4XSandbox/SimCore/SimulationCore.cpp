#include "SimulationCore.h"
#include "EvoRng.h"
#include "SpeciesData.h"
#include "BiomeData.h"
#include "Evo4XSandbox.h"

// ============================================================================
// SimulateTick — full tick, two-phase
// ============================================================================

void FSimulationCore::SimulateTick(FWorldState& World)
{
	// Clear migration buffers
	World.PendingMigrants.Empty();
	World.MigrationEdgesThisTick.Empty();

	// ── PHASE 1: per-region calculations (all regions before phase 2) ──
	for (FRegion& Region : World.Regions)
	{
		Region.TickBreakdown.Empty();

		// Step 1: Apply active effects
		ApplyActiveEffects(Region);

		// Step 2: Births
		ComputeBirths(Region);

		// Step 3: Resource allocation
		AllocateResources(Region);

		// Steps 4-5: Starving newborns + outgoing migration
		ComputeMigration(World, Region);

		// Step 6: Local starvation commit
		CommitStarvation(Region, World.Config);
	}

	// ── PHASE 2: global ──

	// Step 7: Apply incoming migrants
	ApplyIncomingMigrants(World);

	// Step 8: Extinction cleanup
	ExtinctionCleanup(World);

	// Tick down effects
	for (FRegion& Region : World.Regions)
	{
		TickDownEffects(Region);
	}

	// Step 9: Metrics / trends / history
	UpdateMetrics(World);

	// Step 10: Advance tick
	World.CurrentTick++;
}

// ============================================================================
// Step 1: Apply active effects
// ============================================================================

void FSimulationCore::ApplyActiveEffects(FRegion& Region)
{
	// Start from base resources
	Region.EffR1 = Region.BaseR1;
	Region.EffR2 = Region.BaseR2;
	Region.EffR3 = Region.BaseR3;

	// Apply multipliers from active effects
	for (const FActiveEffect& Eff : Region.ActiveEffects)
	{
		Region.EffR1 *= Eff.MultR1;
		Region.EffR2 *= Eff.MultR2;
		Region.EffR3 *= Eff.MultR3;
	}
}

// ============================================================================
// Step 2: Births
// ============================================================================

void FSimulationCore::ComputeBirths(FRegion& Region)
{
	for (auto& Pair : Region.Populations)
	{
		const FName& SpeciesId = Pair.Key;
		const float Pop = Pair.Value;
		const FSpeciesData& Spec = FSpeciesRegistry::Get(SpeciesId);

		FSpeciesInRegion& Breakdown = Region.TickBreakdown.FindOrAdd(SpeciesId);
		const float Births = Pop * Spec.R;
		Breakdown.Population = Pop;
		Breakdown.PopPreBirth = Pop + Births;
	}
}

// ============================================================================
// Step 3: Resource allocation
// ============================================================================

void FSimulationCore::AllocateResources(FRegion& Region)
{
	// Initialize food allocated to 0 for all species
	for (auto& Pair : Region.TickBreakdown)
	{
		Pair.Value.FoodAllocated = 0.f;
	}

	// Available resources this tick
	const float AvailResources[3] = { Region.EffR1, Region.EffR2, Region.EffR3 };

	// For each resource, divide proportionally by weight
	for (int32 ResIdx = 0; ResIdx < 3; ++ResIdx)
	{
		const float Avail = AvailResources[ResIdx];
		if (Avail <= 0.f) continue;

		// Compute weights: w[species] = n_pre * b[resource]
		float TotalWeight = 0.f;
		TArray<TPair<FName, float>> Weights;

		for (const auto& Pair : Region.TickBreakdown)
		{
			const FName& SpeciesId = Pair.Key;
			const float NPre = Pair.Value.PopPreBirth;
			const FSpeciesData& Spec = FSpeciesRegistry::Get(SpeciesId);

			float B = 0.f;
			switch (ResIdx)
			{
			case 0: B = Spec.BR1; break;
			case 1: B = Spec.BR2; break;
			case 2: B = Spec.BR3; break;
			}

			const float W = NPre * B;
			if (W > 0.f)
			{
				Weights.Emplace(SpeciesId, W);
				TotalWeight += W;
			}
		}

		if (TotalWeight <= 0.f) continue;

		// Allocate proportionally
		for (const auto& WPair : Weights)
		{
			const float Share = Avail * WPair.Value / TotalWeight;
			Region.TickBreakdown[WPair.Key].FoodAllocated += Share;
		}
	}
}

// ============================================================================
// Steps 4-5: Starving newborns + outgoing migration
// ============================================================================

void FSimulationCore::ComputeMigration(FWorldState& World, FRegion& Region)
{
	const FString& Seed = World.Config.WorldSeed;
	const int32 Tick = World.CurrentTick;

	for (auto& Pair : Region.TickBreakdown)
	{
		const FName& SpeciesId = Pair.Key;
		FSpeciesInRegion& Bd = Pair.Value;
		const FSpeciesData& Spec = FSpeciesRegistry::Get(SpeciesId);

		// Step 4: Identify starving newborns
		const float Births = Bd.PopPreBirth - Bd.Population; // = pop * r
		Bd.StarvingTotal = FMath::Max(0.f, Bd.PopPreBirth - Bd.FoodAllocated);
		Bd.StarvingNewborns = FMath::Min(Births, Bd.StarvingTotal);

		// Step 5: Compute outgoing expeditions
		const int32 StarvingPairs = FMath::FloorToInt32(Bd.StarvingNewborns / 2.f);

		if (StarvingPairs <= 0 || Region.Neighbors.Num() == 0)
		{
			Bd.OutgoingMigrants = 0.f;
			continue;
		}

		// Bernoulli loop: each pair migrates with probability radiation/100
		const float RadiationChance = Spec.Radiation / 100.f;
		int32 NumExpeditions = 0;

		for (int32 PairIdx = 0; PairIdx < StarvingPairs; ++PairIdx)
		{
			const FString Key = FString::Printf(TEXT("migrate|%d|%d|%s|%d"),
				Tick, Region.RegionId, *SpeciesId.ToString(), PairIdx);
			const float Roll = FEvoRng::Rand01(Seed, Key);
			if (Roll < RadiationChance)
			{
				NumExpeditions++;
			}
		}

		if (NumExpeditions == 0)
		{
			Bd.OutgoingMigrants = 0.f;
			continue;
		}

		Bd.OutgoingMigrants = static_cast<float>(NumExpeditions * 2);

		// Distribute expeditions among neighbors
		const int32 N = Region.Neighbors.Num();
		const int32 Base = NumExpeditions / N;
		const int32 Rem = NumExpeditions % N;

		// Give base to everyone
		for (int32 i = 0; i < N; ++i)
		{
			const int32 NeighborId = Region.Neighbors[i];
			const float Migrants = static_cast<float>(Base * 2);
			if (Migrants > 0.f)
			{
				World.PendingMigrants.FindOrAdd(NeighborId).FindOrAdd(SpeciesId) += Migrants;

				FMigrationEdge Edge;
				Edge.FromRegion = Region.RegionId;
				Edge.ToRegion = NeighborId;
				Edge.SpeciesId = SpeciesId;
				Edge.Count = Migrants;
				World.MigrationEdgesThisTick.Add(Edge);
			}
		}

		// Distribute remainder: pick Rem neighbors deterministically
		if (Rem > 0)
		{
			const FString ShuffleKey = FString::Printf(TEXT("mig_rem|%d|%d|%s"),
				Tick, Region.RegionId, *SpeciesId.ToString());
			TArray<int32> ShuffledIdx = FEvoRng::ShuffleIndices(Seed, ShuffleKey, N);

			for (int32 i = 0; i < Rem; ++i)
			{
				const int32 NeighborId = Region.Neighbors[ShuffledIdx[i]];
				World.PendingMigrants.FindOrAdd(NeighborId).FindOrAdd(SpeciesId) += 2.f;

				// Find existing edge or add new one
				bool bFound = false;
				for (FMigrationEdge& Edge : World.MigrationEdgesThisTick)
				{
					if (Edge.FromRegion == Region.RegionId
						&& Edge.ToRegion == NeighborId
						&& Edge.SpeciesId == SpeciesId)
					{
						Edge.Count += 2.f;
						bFound = true;
						break;
					}
				}
				if (!bFound)
				{
					FMigrationEdge Edge;
					Edge.FromRegion = Region.RegionId;
					Edge.ToRegion = NeighborId;
					Edge.SpeciesId = SpeciesId;
					Edge.Count = 2.f;
					World.MigrationEdgesThisTick.Add(Edge);
				}
			}
		}
	}
}

// ============================================================================
// Step 6: Commit local starvation
// ============================================================================

void FSimulationCore::CommitStarvation(FRegion& Region, const FSimConfig& Config)
{
	for (auto& Pair : Region.TickBreakdown)
	{
		const FName& SpeciesId = Pair.Key;
		FSpeciesInRegion& Bd = Pair.Value;

		// Survivors = min(n_pre, food_allocated)
		// But also subtract outgoing migrants from n_pre before capping
		const float EffectiveNPre = Bd.PopPreBirth - Bd.OutgoingMigrants;
		const float Survivors = FMath::Min(EffectiveNPre, Bd.FoodAllocated);

		Region.Populations[SpeciesId] = FMath::Max(0.f, Survivors);
	}
}

// ============================================================================
// Step 7: Apply incoming migrants
// ============================================================================

void FSimulationCore::ApplyIncomingMigrants(FWorldState& World)
{
	for (const auto& RegionEntry : World.PendingMigrants)
	{
		const int32 RegionId = RegionEntry.Key;
		if (!World.Regions.IsValidIndex(RegionId)) continue;

		FRegion& Region = World.Regions[RegionId];

		for (const auto& SpeciesEntry : RegionEntry.Value)
		{
			const FName& SpeciesId = SpeciesEntry.Key;
			const float Incoming = SpeciesEntry.Value;

			Region.Populations.FindOrAdd(SpeciesId) += Incoming;

			// Record in breakdown for inspector
			FSpeciesInRegion& Bd = Region.TickBreakdown.FindOrAdd(SpeciesId);
			Bd.IncomingMigrants += Incoming;
		}
	}
}

// ============================================================================
// Step 8: Extinction cleanup
// ============================================================================

void FSimulationCore::ExtinctionCleanup(FWorldState& World)
{
	for (FRegion& Region : World.Regions)
	{
		TArray<FName> ToRemove;
		for (const auto& Pair : Region.Populations)
		{
			if (Pair.Value <= World.Config.ExtinctionEps)
			{
				ToRemove.Add(Pair.Key);
			}
		}
		for (const FName& Id : ToRemove)
		{
			Region.Populations.Remove(Id);
			Region.TickBreakdown.Remove(Id);
		}
	}
}

// ============================================================================
// Step 9: Metrics / trends / history
// ============================================================================

void FSimulationCore::UpdateMetrics(FWorldState& World)
{
	const int32 HistLen = World.Config.TrendHistoryLen;
	const int32 TrendWindow = World.Config.TrendWindowShort;
	const float TrendThresh = World.Config.TrendThreshold;

	// Collect global species totals
	World.Metrics.SpeciesTotalPop.Empty();
	World.Metrics.TotalWorldBiomass = 0.f;

	for (FRegion& Region : World.Regions)
	{
		// Update region totals
		Region.TotalBiomass = Region.GetTotalPopulation();
		Region.DominantSpecies = Region.FindDominantSpecies();

		// Per-species history in region
		for (const auto& Pair : Region.Populations)
		{
			const FName& SpeciesId = Pair.Key;
			const float Pop = Pair.Value;

			// Ring buffer
			TArray<float>& Hist = Region.PopHistory.FindOrAdd(SpeciesId);
			Hist.Add(Pop);
			if (Hist.Num() > HistLen)
			{
				Hist.RemoveAt(0, Hist.Num() - HistLen);
			}

			// Trend for this species in this region
			if (Region.TickBreakdown.Contains(SpeciesId))
			{
				FSpeciesInRegion& Bd = Region.TickBreakdown[SpeciesId];
				if (Hist.Num() > TrendWindow)
				{
					const float OldPop = Hist[Hist.Num() - 1 - TrendWindow];
					if (OldPop > 0.f)
					{
						const float Delta = (Pop - OldPop) / OldPop;
						if (Delta > TrendThresh) Bd.Trend = ETrend::Up;
						else if (Delta < -TrendThresh) Bd.Trend = ETrend::Down;
						else Bd.Trend = ETrend::Flat;
					}
					else
					{
						Bd.Trend = Pop > 0.f ? ETrend::Up : ETrend::Flat;
					}
				}
			}

			// Global totals
			World.Metrics.SpeciesTotalPop.FindOrAdd(SpeciesId) += Pop;
			World.Metrics.TotalWorldBiomass += Pop;
		}

		// Clean up history for extinct species
		TArray<FName> HistToRemove;
		for (const auto& HistPair : Region.PopHistory)
		{
			if (!Region.Populations.Contains(HistPair.Key))
			{
				HistToRemove.Add(HistPair.Key);
			}
		}
		for (const FName& Id : HistToRemove)
		{
			Region.PopHistory.Remove(Id);
		}

		// Region biomass trend
		// Use sum of all species pops over time
		float CurrentTotal = Region.TotalBiomass;
		float PastTotal = 0.f;
		bool bHasPast = false;
		for (const auto& HistPair : Region.PopHistory)
		{
			const TArray<float>& Hist = HistPair.Value;
			if (Hist.Num() > TrendWindow)
			{
				PastTotal += Hist[Hist.Num() - 1 - TrendWindow];
				bHasPast = true;
			}
		}
		if (bHasPast && PastTotal > 0.f)
		{
			const float Delta = (CurrentTotal - PastTotal) / PastTotal;
			if (Delta > TrendThresh) Region.BiomasssTrend = ETrend::Up;
			else if (Delta < -TrendThresh) Region.BiomasssTrend = ETrend::Down;
			else Region.BiomasssTrend = ETrend::Flat;
		}
	}

	// Global pop history
	for (const auto& Pair : World.Metrics.SpeciesTotalPop)
	{
		TArray<float>& GHist = World.GlobalPopHistory.FindOrAdd(Pair.Key);
		GHist.Add(Pair.Value);
		if (GHist.Num() > HistLen)
		{
			GHist.RemoveAt(0, GHist.Num() - HistLen);
		}
	}

	// Count alive species
	World.Metrics.AliveSpeciesCount = World.Metrics.SpeciesTotalPop.Num();

	// Find fastest growing / declining (by % change over last few ticks)
	float BestGrowth = -1.f;
	float WorstDecline = 1.f;
	World.Metrics.FastestGrowing = FName();
	World.Metrics.FastestDeclining = FName();

	for (const auto& Pair : World.GlobalPopHistory)
	{
		const TArray<float>& GHist = Pair.Value;
		if (GHist.Num() > TrendWindow)
		{
			const float OldPop = GHist[GHist.Num() - 1 - TrendWindow];
			const float NewPop = GHist.Last();
			if (OldPop > 0.f)
			{
				const float Delta = (NewPop - OldPop) / OldPop;
				if (Delta > BestGrowth)
				{
					BestGrowth = Delta;
					World.Metrics.FastestGrowing = Pair.Key;
				}
				if (Delta < WorstDecline)
				{
					WorstDecline = Delta;
					World.Metrics.FastestDeclining = Pair.Key;
				}
			}
		}
	}
}

// ============================================================================
// Tick down effects
// ============================================================================

void FSimulationCore::TickDownEffects(FRegion& Region)
{
	for (int32 i = Region.ActiveEffects.Num() - 1; i >= 0; --i)
	{
		Region.ActiveEffects[i].TTL--;
		if (Region.ActiveEffects[i].TTL <= 0)
		{
			Region.ActiveEffects.RemoveAt(i);
		}
	}
}
