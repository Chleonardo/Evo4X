#pragma once

#include "CoreMinimal.h"

/**
 * Deterministic stateless RNG based on SHA-256.
 * Identical algorithm to the Python Evo4X engine:
 *   rand01(seed, key) = sha256("{seed}|{key}")[:8] as uint32 / 2^32
 *
 * Same seed + key always produces the same result, across platforms.
 */
class EVO4XSANDBOX_API FEvoRng
{
public:
	/**
	 * Returns a deterministic float in [0, 1) for the given seed and key.
	 * Matches Python: rand01(seed, key) = sha256(f"{seed}|{key}")[:8] / 2^32
	 */
	static float Rand01(const FString& Seed, const FString& Key);

	/**
	 * Returns a deterministic index in [0, N-1].
	 * Matches Python: choose_index(seed, key, n) = min(n-1, floor(rand01 * n))
	 */
	static int32 ChooseIndex(const FString& Seed, const FString& Key, int32 N);

	/**
	 * Convenience: Rand01 with a compound key built from multiple parts.
	 * E.g., Rand01(Seed, "migration", RegionId, SpeciesName, PairIndex)
	 */
	static float Rand01(const FString& Seed, const FString& Context, int32 A);
	static float Rand01(const FString& Seed, const FString& Context, int32 A, int32 B);
	static float Rand01(const FString& Seed, const FString& Context, int32 A, const FString& B);

	static int32 ChooseIndex(const FString& Seed, const FString& Context, int32 A, int32 N);

	/**
	 * Deterministic shuffle of indices [0..N-1] using Fisher-Yates with keyed RNG.
	 */
	static TArray<int32> ShuffleIndices(const FString& Seed, const FString& Key, int32 N);
};
