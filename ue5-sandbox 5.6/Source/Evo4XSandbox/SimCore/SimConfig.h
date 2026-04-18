#pragma once

#include "CoreMinimal.h"

/**
 * All tunable simulation parameters in one place.
 */
struct FSimConfig
{
	// ── Map ──
	int32 TotalHexes        = 500;   // approximate land hex count
	int32 NumRegions        = 20;
	int32 MinRegionSize     = 7;     // minimum hexes per region

	// ── Resources ──
	float RichnessPerHex    = 10.f;  // total resource budget per land hex

	// ── Population ──
	float StartPopDensity   = 2.f;   // start pop per species = hex_count * this
	float ExtinctionEps     = 1.f;   // species dies in region if pop <= this

	// ── Simulation ──
	float TickSpeedSlow     = 5.f;   // seconds per tick (slow / observation mode)
	float TickSpeedFast     = 1.f;   // seconds per tick (fast mode)
	float TickSpeedVeryFast = 0.5f;  // seconds per tick (very fast = 2 ticks/sec)
	bool  bEnableEvents     = false; // fluctuation events (architecture-ready, off in v1)

	// ── Trends ──
	int32 TrendHistoryLen   = 50;    // ticks of history to keep per region
	int32 TrendWindowShort  = 3;     // ticks to compare for trend arrow (up/down/flat)
	float TrendThreshold    = 0.05f; // ±5% change = flat

	// ── Seeds ──
	FString WorldSeed       = TEXT("42");
};
