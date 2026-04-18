#pragma once

#include "CoreMinimal.h"

/**
 * Biome archetype — defines resource ratios and visual color for a region type.
 * 9 archetypes from Evo4X savanna setting.
 */
struct FBiomeData
{
	FName Id;
	FString UiName;

	// Resource ratios (sum to ~100, normalized when used)
	float RatioR1 = 0.f; // Grass
	float RatioR2 = 0.f; // Leaves
	float RatioR3 = 0.f; // Roots

	// Map color (linear)
	FLinearColor Color;

	// NPC packs: each pack is a list of species IDs that can spawn together
	TArray<TArray<FName>> Packs;

	// Normalized ratios (computed once)
	float NormR1() const { const float S = RatioR1 + RatioR2 + RatioR3; return S > 0.f ? RatioR1 / S : 0.f; }
	float NormR2() const { const float S = RatioR1 + RatioR2 + RatioR3; return S > 0.f ? RatioR2 / S : 0.f; }
	float NormR3() const { const float S = RatioR1 + RatioR2 + RatioR3; return S > 0.f ? RatioR3 / S : 0.f; }
};

/**
 * Static registry of all biome archetypes.
 */
class EVO4XSANDBOX_API FBiomeRegistry
{
public:
	static void Initialize();
	static const FBiomeData& Get(FName BiomeId);
	static const TArray<FName>& GetAllIds();
	static int32 Num();

private:
	static TMap<FName, FBiomeData> Biomes;
	static TArray<FName> AllIds;
	static bool bInitialized;

	static void Add(const FName& Id, const FString& UiName,
	                float R1, float R2, float R3,
	                const FLinearColor& Color,
	                const TArray<TArray<FName>>& Packs);
};
