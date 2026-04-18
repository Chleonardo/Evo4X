#include "BiomeData.h"

TMap<FName, FBiomeData> FBiomeRegistry::Biomes;
TArray<FName> FBiomeRegistry::AllIds;
bool FBiomeRegistry::bInitialized = false;

void FBiomeRegistry::Add(const FName& Id, const FString& UiName,
                         float R1, float R2, float R3,
                         const FLinearColor& Color,
                         const TArray<TArray<FName>>& Packs)
{
	FBiomeData B;
	B.Id = Id;
	B.UiName = UiName;
	B.RatioR1 = R1;
	B.RatioR2 = R2;
	B.RatioR3 = R3;
	B.Color = Color;
	B.Packs = Packs;
	Biomes.Add(Id, MoveTemp(B));
	AllIds.Add(Id);
}

void FBiomeRegistry::Initialize()
{
	if (bInitialized) return;
	bInitialized = true;

	// Colors:
	// Grass-heavy  = light green
	// Leaf-heavy   = dark green
	// Root-heavy   = ochre / yellow-brown
	// Mixed        = interpolated

	// ── grassland: R1=100, R2=0, R3=0 ──
	Add(TEXT("grassland"), TEXT("Grassland"), 100, 0, 0,
		FLinearColor(0.35f, 0.90f, 0.10f), // vivid lime green
		{
			{},
			{FName("gazelle")},
			{FName("zebra")},
			{FName("buffalo")}
		});

	// ── open_savanna: R1=67, R2=33, R3=0 ──
	Add(TEXT("open_savanna"), TEXT("Open Savanna"), 67, 33, 0,
		FLinearColor(0.85f, 0.78f, 0.15f), // golden yellow
		{
			{FName("giraffe")},
			{FName("giraffe"), FName("gazelle")},
			{FName("giraffe"), FName("zebra")},
			{FName("giraffe"), FName("buffalo")},
			{FName("giraffe"), FName("warthog")},
			{FName("giraffe"), FName("impala")},
			{FName("impala")}
		});

	// ── woodland: R1=33, R2=67, R3=0 ──
	Add(TEXT("woodland"), TEXT("Woodland"), 33, 67, 0,
		FLinearColor(0.08f, 0.50f, 0.08f), // deep forest green
		{
			{FName("giraffe")},
			{FName("giraffe"), FName("buffalo")},
			{FName("giraffe"), FName("impala")}
		});

	// ── dense_grove: R1=20, R2=80, R3=0 ──
	Add(TEXT("dense_grove"), TEXT("Dense Grove"), 20, 80, 0,
		FLinearColor(0.03f, 0.28f, 0.03f), // very dark jungle green
		{
			{FName("giraffe")},
			{FName("elephant")}
		});

	// ── root_patch: R1=67, R2=0, R3=33 ──
	Add(TEXT("root_patch"), TEXT("Root Patch"), 67, 0, 33,
		FLinearColor(0.80f, 0.60f, 0.10f), // warm amber
		{
			{FName("mole_rat")},
			{FName("mole_rat"), FName("gazelle")},
			{FName("mole_rat"), FName("zebra")},
			{FName("mole_rat"), FName("buffalo")},
			{FName("mole_rat"), FName("giraffe")},
			{FName("warthog")},
			{FName("warthog"), FName("gazelle")},
			{FName("warthog"), FName("impala")},
			{FName("warthog"), FName("buffalo")},
			{FName("warthog"), FName("elephant")}
		});

	// ── rootland: R1=33, R2=0, R3=67 ──
	Add(TEXT("rootland"), TEXT("Rootland"), 33, 0, 67,
		FLinearColor(0.70f, 0.38f, 0.05f), // burnt orange-brown
		{
			{FName("mole_rat")},
			{FName("mole_rat"), FName("gazelle")},
			{FName("mole_rat"), FName("zebra")},
			{FName("mole_rat"), FName("buffalo")},
			{FName("mole_rat"), FName("giraffe")},
			{FName("mole_rat"), FName("elephant")},
			{FName("warthog")}
		});

	// ── diverse_savanna: R1=50, R2=30, R3=20 ──
	Add(TEXT("diverse_savanna"), TEXT("Diverse Savanna"), 50, 30, 20,
		FLinearColor(0.65f, 0.85f, 0.20f), // yellow-green
		{
			{FName("giraffe"), FName("mole_rat"), FName("gazelle")},
			{FName("giraffe"), FName("mole_rat"), FName("impala")},
			{FName("giraffe"), FName("mole_rat"), FName("buffalo")},
			{FName("giraffe"), FName("warthog"), FName("impala")},
			{FName("giraffe"), FName("warthog"), FName("buffalo")},
			{FName("giraffe"), FName("mole_rat"), FName("zebra")}
		});

	// ── root_mosaic: R1=40, R2=20, R3=40 ──
	Add(TEXT("root_mosaic"), TEXT("Root Mosaic"), 40, 20, 40,
		FLinearColor(0.75f, 0.55f, 0.08f), // deep ochre
		{
			{FName("giraffe"), FName("mole_rat"), FName("gazelle")},
			{FName("giraffe"), FName("mole_rat"), FName("impala")},
			{FName("giraffe"), FName("mole_rat"), FName("buffalo")},
			{FName("giraffe"), FName("warthog"), FName("impala")},
			{FName("giraffe"), FName("warthog"), FName("buffalo")},
			{FName("giraffe"), FName("mole_rat"), FName("zebra")}
		});

	// ── leaf_mosaic: R1=30, R2=50, R3=20 ──
	Add(TEXT("leaf_mosaic"), TEXT("Leaf Mosaic"), 30, 50, 20,
		FLinearColor(0.10f, 0.65f, 0.40f), // teal-green
		{
			{FName("giraffe"), FName("mole_rat"), FName("gazelle")},
			{FName("giraffe"), FName("mole_rat"), FName("impala")},
			{FName("giraffe"), FName("mole_rat"), FName("buffalo")},
			{FName("giraffe"), FName("warthog"), FName("buffalo")}
		});
}

const FBiomeData& FBiomeRegistry::Get(FName BiomeId)
{
	if (!bInitialized) Initialize();
	return Biomes.FindChecked(BiomeId);
}

const TArray<FName>& FBiomeRegistry::GetAllIds()
{
	if (!bInitialized) Initialize();
	return AllIds;
}

int32 FBiomeRegistry::Num()
{
	if (!bInitialized) Initialize();
	return AllIds.Num();
}
