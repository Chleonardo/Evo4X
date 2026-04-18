#include "Region.h"
#include "BiomeData.h"

void FRegion::ComputeBaseResources(float RichnessPerHex)
{
	const FBiomeData& Biome = FBiomeRegistry::Get(BiomeId);
	const float TotalRichness = HexCount * RichnessPerHex;
	BaseR1 = TotalRichness * Biome.NormR1();
	BaseR2 = TotalRichness * Biome.NormR2();
	BaseR3 = TotalRichness * Biome.NormR3();
	EffR1 = BaseR1;
	EffR2 = BaseR2;
	EffR3 = BaseR3;
}

float FRegion::GetTotalPopulation() const
{
	float Total = 0.f;
	for (const auto& Pair : Populations)
	{
		Total += Pair.Value;
	}
	return Total;
}

FName FRegion::FindDominantSpecies() const
{
	FName Best;
	float BestPop = 0.f;
	for (const auto& Pair : Populations)
	{
		if (Pair.Value > BestPop)
		{
			BestPop = Pair.Value;
			Best = Pair.Key;
		}
	}
	return Best;
}
