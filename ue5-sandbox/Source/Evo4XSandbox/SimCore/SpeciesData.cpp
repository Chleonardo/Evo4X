#include "SpeciesData.h"

TMap<FName, FSpeciesData> FSpeciesRegistry::Species;
TArray<FName> FSpeciesRegistry::AllIds;
bool FSpeciesRegistry::bInitialized = false;

void FSpeciesRegistry::Add(const FName& Id, const FString& UiName, const FString& DisplayChar,
                           float InR, float InBR1, float InBR2, float InBR3,
                           float InRadiation, const FLinearColor& Color)
{
	FSpeciesData S;
	S.Id = Id;
	S.UiName = UiName;
	S.DisplayChar = DisplayChar;
	S.R = InR;
	S.BR1 = InBR1;
	S.BR2 = InBR2;
	S.BR3 = InBR3;
	S.Radiation = InRadiation;
	S.Color = Color;
	Species.Add(Id, MoveTemp(S));
	AllIds.Add(Id);
}

void FSpeciesRegistry::Initialize()
{
	if (bInitialized) return;
	bInitialized = true;

	//                  Id              UiName       Char   r     bR1  bR2  bR3   rad%   Color
	Add(TEXT("gazelle"),  TEXT("Gazelle"),  TEXT("Gz"), 1.00f, 1.0f, 0.0f, 0.0f,  8.f, FLinearColor(0.95f, 0.85f, 0.40f)); // gold
	Add(TEXT("impala"),   TEXT("Impala"),   TEXT("Im"), 0.95f, 1.2f, 0.6f, 0.0f,  6.f, FLinearColor(0.85f, 0.55f, 0.25f)); // orange-brown
	Add(TEXT("zebra"),    TEXT("Zebra"),    TEXT("Zb"), 0.75f, 1.1f, 0.0f, 0.0f,  5.f, FLinearColor(0.90f, 0.90f, 0.90f)); // white/gray
	Add(TEXT("buffalo"),  TEXT("Buffalo"),  TEXT("Bf"), 0.65f, 2.0f, 0.0f, 0.0f,  3.f, FLinearColor(0.40f, 0.25f, 0.15f)); // dark brown
	Add(TEXT("giraffe"),  TEXT("Giraffe"),  TEXT("Gi"), 0.40f, 0.5f, 3.0f, 0.0f,  4.f, FLinearColor(0.90f, 0.75f, 0.30f)); // yellow-spotted
	Add(TEXT("elephant"), TEXT("Elephant"), TEXT("El"), 0.12f, 2.0f, 3.0f, 0.0f,  2.f, FLinearColor(0.55f, 0.55f, 0.55f)); // gray
	Add(TEXT("warthog"),  TEXT("Warthog"),  TEXT("Wh"), 1.20f, 0.5f, 0.0f, 2.0f,  6.f, FLinearColor(0.65f, 0.40f, 0.30f)); // reddish brown
	Add(TEXT("mole_rat"), TEXT("Mole Rat"), TEXT("Mr"), 1.80f, 0.0f, 0.0f, 1.0f,  5.f, FLinearColor(0.75f, 0.60f, 0.45f)); // sandy
}

const FSpeciesData& FSpeciesRegistry::Get(FName SpeciesId)
{
	if (!bInitialized) Initialize();
	return Species.FindChecked(SpeciesId);
}

const TArray<FName>& FSpeciesRegistry::GetAllIds()
{
	if (!bInitialized) Initialize();
	return AllIds;
}

int32 FSpeciesRegistry::Num()
{
	if (!bInitialized) Initialize();
	return AllIds.Num();
}
