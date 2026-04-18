#pragma once

#include "CoreMinimal.h"

/**
 * Species definition — stats, resource affinity, and radiation (migration tendency).
 * 8 savanna species from Evo4X.
 */
struct FSpeciesData
{
	FName Id;
	FString UiName;
	FString DisplayChar; // 1-2 char abbreviation for map rendering (v1, no sprites)

	// Growth rate per tick: births = pop * r
	float R = 0.f;

	// Resource affinity weights
	float BR1 = 0.f; // Grass
	float BR2 = 0.f; // Leaves
	float BR3 = 0.f; // Roots

	// Radiation: % chance each starving pair attempts migration (0-100)
	float Radiation = 0.f;

	// Map color for species overlays
	FLinearColor Color;
};

/**
 * Static registry of all species.
 */
class EVO4XSANDBOX_API FSpeciesRegistry
{
public:
	static void Initialize();
	static const FSpeciesData& Get(FName SpeciesId);
	static const TArray<FName>& GetAllIds();
	static int32 Num();

private:
	static TMap<FName, FSpeciesData> Species;
	static TArray<FName> AllIds;
	static bool bInitialized;

	static void Add(const FName& Id, const FString& UiName, const FString& DisplayChar,
	                float InR, float InBR1, float InBR2, float InBR3,
	                float InRadiation, const FLinearColor& Color);
};
