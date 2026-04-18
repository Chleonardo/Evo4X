#include "SandboxController.h"
#include "SimCore/SimulationCore.h"
#include "MapGen/WorldGenerator.h"
#include "Evo4XSandbox.h"

ASandboxController::ASandboxController()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ASandboxController::BeginPlay()
{
	Super::BeginPlay();
	InitializeWorld();
}

void ASandboxController::InitializeWorld()
{
	FSimConfig Config;
	Config.WorldSeed = WorldSeed;
	Config.TotalHexes = TotalHexes;
	Config.NumRegions = NumRegions;
	Config.RichnessPerHex = RichnessPerHex;
	Config.StartPopDensity = StartPopDensity;

	FWorldGenerator::Generate(Config, HexGrid, WorldState);

	CurrentTick = 0;
	TickAccumulator = 0.f;
	bIsPaused = true;
	SpeedMode = 0;
	SelectedRegionId = -1;

	UE_LOG(LogEvo4X, Log, TEXT("World initialized. %d regions, %d land hexes"),
		WorldState.Regions.Num(), HexGrid.LandIndices.Num());

	OnTickCompleted.Broadcast();
}

void ASandboxController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (bIsPaused || SpeedMode == 0) return;

	const float Interval = GetCurrentTickInterval();
	TickAccumulator += DeltaTime;

	if (TickAccumulator >= Interval)
	{
		TickAccumulator -= Interval;
		StepOneTick();
	}
}

void ASandboxController::StepOneTick()
{
	FSimulationCore::SimulateTick(WorldState);
	CurrentTick = WorldState.CurrentTick;

	OnTickCompleted.Broadcast();
}

void ASandboxController::TogglePause()
{
	if (bIsPaused)
	{
		bIsPaused = false;
		if (SpeedMode == 0) SpeedMode = 1;
	}
	else
	{
		bIsPaused = true;
		SpeedMode = 0;
	}
}

void ASandboxController::SetSpeed(int32 Mode)
{
	SpeedMode = FMath::Clamp(Mode, 0, 2);
	bIsPaused = (SpeedMode == 0);
	TickAccumulator = 0.f;
}

void ASandboxController::SelectRegion(int32 RegionId)
{
	SelectedRegionId = RegionId;
	OnRegionSelected.Broadcast(RegionId);
}

float ASandboxController::GetTotalWorldBiomass() const
{
	return WorldState.Metrics.TotalWorldBiomass;
}

int32 ASandboxController::GetAliveSpeciesCount() const
{
	return WorldState.Metrics.AliveSpeciesCount;
}

float ASandboxController::GetCurrentTickInterval() const
{
	switch (SpeedMode)
	{
	case 1: return WorldState.Config.TickSpeedSlow; // 5 sec
	case 2: return WorldState.Config.TickSpeedFast; // 1 sec
	default: return 0.f;
	}
}
