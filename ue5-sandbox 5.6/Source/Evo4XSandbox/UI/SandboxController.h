#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SimCore/WorldState.h"
#include "MapGen/HexGrid.h"
#include "SandboxController.generated.h"

class AHexMapRenderer;
class ARegionOverlay;
class AMigrationArrowManager;

/**
 * Main controller for the sandbox simulation.
 * Manages world state, tick loop, speed/pause, and coordinates rendering.
 */
UCLASS()
class EVO4XSANDBOX_API ASandboxController : public AActor
{
	GENERATED_BODY()

public:
	ASandboxController();

	// ── Configuration ──

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation")
	FString WorldSeed = TEXT("42");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation")
	int32 TotalHexes = 500;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation")
	int32 NumRegions = 20;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation")
	float RichnessPerHex = 10.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation")
	float StartPopDensity = 2.f;

	// ── Runtime state ──

	UPROPERTY(BlueprintReadOnly, Category = "Simulation")
	int32 CurrentTick = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Simulation")
	bool bIsPaused = true;

	/** 0 = paused, 1 = slow, 2 = fast */
	UPROPERTY(BlueprintReadOnly, Category = "Simulation")
	int32 SpeedMode = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Simulation")
	int32 SelectedRegionId = -1;

	// ── Blueprint-callable ──

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void InitializeWorld();

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void TogglePause();

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void SetSpeed(int32 Mode);

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void SelectRegion(int32 RegionId);

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void StepOneTick();

	// ── Accessors for UI ──

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	float GetTotalWorldBiomass() const;

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	int32 GetAliveSpeciesCount() const;

	/** Get the world state (for C++ access). */
	const FWorldState& GetWorldState() const { return WorldState; }
	const FHexGrid& GetHexGrid() const { return HexGrid; }

	/** Delegate: fired after each tick for UI refresh. */
	DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnTickCompleted);

	UPROPERTY(BlueprintAssignable, Category = "Simulation")
	FOnTickCompleted OnTickCompleted;

	DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRegionSelected, int32, RegionId);

	UPROPERTY(BlueprintAssignable, Category = "Simulation")
	FOnRegionSelected OnRegionSelected;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	FWorldState WorldState;
	FHexGrid HexGrid;

	float TickAccumulator = 0.f;

	float GetCurrentTickInterval() const;
	void SyncSpeed();
};
