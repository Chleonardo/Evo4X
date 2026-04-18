#include "SandboxHUD.h"
#include "Engine/Canvas.h"
#include "Engine/Font.h"

TCHAR ASandboxHUD::TrendChar(ETrend T)
{
	switch (T)
	{
	case ETrend::Up:   return TEXT('^');
	case ETrend::Down: return TEXT('v');
	default:           return TEXT('-');
	}
}

FString ASandboxHUD::SpeedString() const
{
	if (!CachedWorld) return TEXT("--");
	switch (CachedWorld->Speed)
	{
	case ESimSpeed::Paused: return TEXT("PAUSED");
	case ESimSpeed::Slow:   return TEXT("SLOW (1t/5s)");
	case ESimSpeed::Fast:   return TEXT("FAST (1t/1s)");
	default:                return TEXT("--");
	}
}

void ASandboxHUD::DrawTextShadow(const FString& Text, float X, float Y,
	FLinearColor Color, float Scale)
{
	// Shadow
	DrawText(Text, FLinearColor(0.f, 0.f, 0.f, 0.8f),
		X + 1.f, Y + 1.f, nullptr, Scale, false);
	// Main
	DrawText(Text, Color, X, Y, nullptr, Scale, false);
}

void ASandboxHUD::DrawHUD()
{
	Super::DrawHUD();

	if (!CachedWorld) return;

	// Global panel: top-left
	DrawGlobalPanel(20.f, 20.f);

	// Controls hint: bottom-left
	DrawControlsHint(20.f, Canvas->ClipY - 120.f);

	// Region inspector: right side
	if (SelectedRegionId >= 0 && CachedWorld->Regions.IsValidIndex(SelectedRegionId))
	{
		DrawRegionInspector(Canvas->ClipX - 350.f, 20.f);
	}

	// History graph: bottom-left
	DrawHistoryGraph(20.f, Canvas->ClipY - 320.f, 400.f, 180.f);
}

// ============================================================================
// Global Panel
// ============================================================================

void ASandboxHUD::DrawGlobalPanel(float X, float Y)
{
	// Background
	DrawRect(FLinearColor(0.f, 0.f, 0.f, 0.7f), X - 5.f, Y - 5.f, 280.f, 170.f);

	float CurY = Y;
	const float LineH = 22.f;

	DrawTextShadow(FString::Printf(TEXT("Tick: %d"), CachedWorld->CurrentTick),
		X, CurY, FLinearColor::White, 1.2f);
	CurY += LineH;

	DrawTextShadow(FString::Printf(TEXT("Speed: %s"), *SpeedString()),
		X, CurY, FLinearColor::Yellow, 1.0f);
	CurY += LineH;

	DrawTextShadow(FString::Printf(TEXT("World Biomass: %.0f"), CachedWorld->Metrics.TotalWorldBiomass),
		X, CurY, FLinearColor::Green, 1.0f);
	CurY += LineH;

	DrawTextShadow(FString::Printf(TEXT("Alive Species: %d"), CachedWorld->Metrics.AliveSpeciesCount),
		X, CurY, FLinearColor::White, 1.0f);
	CurY += LineH;

	if (!CachedWorld->Metrics.FastestGrowing.IsNone())
	{
		const FSpeciesData& Gr = FSpeciesRegistry::Get(CachedWorld->Metrics.FastestGrowing);
		DrawTextShadow(FString::Printf(TEXT("Growing: %s"), *Gr.UiName),
			X, CurY, FLinearColor(0.3f, 1.f, 0.3f), 1.0f);
	}
	CurY += LineH;

	if (!CachedWorld->Metrics.FastestDeclining.IsNone())
	{
		const FSpeciesData& Dc = FSpeciesRegistry::Get(CachedWorld->Metrics.FastestDeclining);
		DrawTextShadow(FString::Printf(TEXT("Declining: %s"), *Dc.UiName),
			X, CurY, FLinearColor(1.f, 0.3f, 0.3f), 1.0f);
	}
}

// ============================================================================
// Region Inspector
// ============================================================================

void ASandboxHUD::DrawRegionInspector(float X, float Y)
{
	const FRegion& Region = CachedWorld->Regions[SelectedRegionId];
	const FBiomeData& Biome = FBiomeRegistry::Get(Region.BiomeId);

	// Background
	DrawRect(FLinearColor(0.f, 0.f, 0.f, 0.75f), X - 5.f, Y - 5.f, 340.f, 500.f);

	float CurY = Y;
	const float LineH = 20.f;

	// Header
	DrawTextShadow(FString::Printf(TEXT("Region %d: %s"), SelectedRegionId, *Biome.UiName),
		X, CurY, FLinearColor::Yellow, 1.2f);
	CurY += LineH + 4.f;

	DrawTextShadow(FString::Printf(TEXT("Hexes: %d  |  Richness: %.0f"),
		Region.HexCount, Region.HexCount * CachedWorld->Config.RichnessPerHex),
		X, CurY, FLinearColor::White, 0.9f);
	CurY += LineH;

	DrawTextShadow(FString::Printf(TEXT("R1(Grass): %.1f  R2(Leaf): %.1f  R3(Root): %.1f"),
		Region.EffR1, Region.EffR2, Region.EffR3),
		X, CurY, FLinearColor(0.8f, 0.8f, 0.6f), 0.9f);
	CurY += LineH;

	DrawTextShadow(FString::Printf(TEXT("Total Biomass: %.0f  %c"),
		Region.TotalBiomass, TrendChar(Region.BiomasssTrend)),
		X, CurY, FLinearColor::Green, 1.0f);
	CurY += LineH + 8.f;

	// Active effects
	if (Region.ActiveEffects.Num() > 0)
	{
		DrawTextShadow(TEXT("Active Effects:"), X, CurY, FLinearColor(1.f, 0.6f, 0.2f), 0.9f);
		CurY += LineH;
		for (const FActiveEffect& Eff : Region.ActiveEffects)
		{
			DrawTextShadow(FString::Printf(TEXT("  %s (TTL %d) R1:%.0f%% R2:%.0f%% R3:%.0f%%"),
				*Eff.EffectType.ToString(), Eff.TTL,
				Eff.MultR1 * 100.f, Eff.MultR2 * 100.f, Eff.MultR3 * 100.f),
				X, CurY, FLinearColor(1.f, 0.8f, 0.4f), 0.8f);
			CurY += LineH;
		}
	}
	else
	{
		DrawTextShadow(TEXT("No active effects"), X, CurY, FLinearColor(0.5f, 0.5f, 0.5f), 0.8f);
		CurY += LineH;
	}

	CurY += 4.f;

	// Species list
	DrawTextShadow(TEXT("Species:"), X, CurY, FLinearColor::White, 1.0f);
	CurY += LineH;

	// Header row
	DrawTextShadow(TEXT("Name        Pop    Share  Trend  Out   In"),
		X, CurY, FLinearColor(0.7f, 0.7f, 0.7f), 0.8f);
	CurY += LineH;

	for (const auto& Pair : Region.Populations)
	{
		const FName& SpeciesId = Pair.Key;
		const float Pop = Pair.Value;
		const FSpeciesData& Spec = FSpeciesRegistry::Get(SpeciesId);

		float Share = Region.TotalBiomass > 0.f ? (Pop / Region.TotalBiomass * 100.f) : 0.f;
		ETrend Trend = ETrend::Flat;
		float Outgoing = 0.f;
		float Incoming = 0.f;

		if (const FSpeciesInRegion* Bd = Region.TickBreakdown.Find(SpeciesId))
		{
			Trend = Bd->Trend;
			Outgoing = Bd->OutgoingMigrants;
			Incoming = Bd->IncomingMigrants;
		}

		FString Line = FString::Printf(TEXT("%-10s %5.0f  %4.0f%%    %c   %4.0f  %4.0f"),
			*Spec.DisplayChar, Pop, Share, TrendChar(Trend), Outgoing, Incoming);

		DrawTextShadow(Line, X, CurY, Spec.Color, 0.85f);
		CurY += LineH;
	}

	CurY += 8.f;

	// Neighbors
	DrawTextShadow(TEXT("Neighbors:"), X, CurY, FLinearColor::White, 0.9f);
	CurY += LineH;

	FString NeighborStr;
	for (const int32 NId : Region.Neighbors)
	{
		if (!NeighborStr.IsEmpty()) NeighborStr += TEXT(", ");
		if (CachedWorld->Regions.IsValidIndex(NId))
		{
			const FBiomeData& NbBiome = FBiomeRegistry::Get(CachedWorld->Regions[NId].BiomeId);
			NeighborStr += FString::Printf(TEXT("R%d(%s)"), NId, *NbBiome.UiName);
		}
	}
	DrawTextShadow(NeighborStr, X, CurY, FLinearColor(0.6f, 0.8f, 1.f), 0.75f);
}

// ============================================================================
// Controls Hint
// ============================================================================

void ASandboxHUD::DrawControlsHint(float X, float Y)
{
	DrawRect(FLinearColor(0.f, 0.f, 0.f, 0.5f), X - 5.f, Y - 5.f, 320.f, 110.f);

	const float LineH = 18.f;
	DrawTextShadow(TEXT("[Space] Pause/Resume  [1] Slow  [2] Fast"),
		X, Y, FLinearColor(0.7f, 0.7f, 0.7f), 0.85f);
	DrawTextShadow(TEXT("[T] Step one tick  [O] Cycle overlay"),
		X, Y + LineH, FLinearColor(0.7f, 0.7f, 0.7f), 0.85f);
	DrawTextShadow(TEXT("[WASD] Pan  [Scroll] Zoom  [Click] Select"),
		X, Y + LineH * 2, FLinearColor(0.7f, 0.7f, 0.7f), 0.85f);
	DrawTextShadow(TEXT("[Esc] Deselect region"),
		X, Y + LineH * 3, FLinearColor(0.7f, 0.7f, 0.7f), 0.85f);
}

// ============================================================================
// History Graph
// ============================================================================

void ASandboxHUD::DrawHistoryGraph(float X, float Y, float W, float H)
{
	if (CachedWorld->GlobalPopHistory.Num() == 0) return;

	// Background
	DrawRect(FLinearColor(0.f, 0.f, 0.f, 0.6f), X - 5.f, Y - 5.f, W + 10.f, H + 10.f);

	DrawTextShadow(TEXT("Population History"), X, Y - 20.f,
		FLinearColor::White, 0.85f);

	// Find max value for scaling
	float MaxVal = 1.f;
	int32 MaxLen = 0;
	for (const auto& Pair : CachedWorld->GlobalPopHistory)
	{
		for (const float V : Pair.Value)
		{
			MaxVal = FMath::Max(MaxVal, V);
		}
		MaxLen = FMath::Max(MaxLen, Pair.Value.Num());
	}

	if (MaxLen < 2) return;

	// Draw a line per species
	for (const auto& Pair : CachedWorld->GlobalPopHistory)
	{
		const TArray<float>& Hist = Pair.Value;
		if (Hist.Num() < 2) continue;

		const FSpeciesData& Spec = FSpeciesRegistry::Get(Pair.Key);
		const FLinearColor Color = Spec.Color;

		for (int32 i = 1; i < Hist.Num(); ++i)
		{
			const float X0 = X + (static_cast<float>(i - 1) / (MaxLen - 1)) * W;
			const float X1 = X + (static_cast<float>(i) / (MaxLen - 1)) * W;
			const float Y0 = Y + H - (Hist[i - 1] / MaxVal) * H;
			const float Y1 = Y + H - (Hist[i] / MaxVal) * H;

			DrawLine(X0, Y0, X1, Y1, Color.ToFColor(true), 2.f);
		}

		// Label at the end
		if (Hist.Num() > 0)
		{
			const float LabelX = X + W + 5.f;
			const float LabelY = Y + H - (Hist.Last() / MaxVal) * H - 6.f;
			DrawTextShadow(Spec.DisplayChar, LabelX, LabelY, Color, 0.7f);
		}
	}
}
