#include "Evo4XSandboxGameMode.h"
#include "UI/SandboxController.h"
#include "UI/HexMapRenderer.h"
#include "UI/RegionOverlay.h"
#include "UI/MigrationArrowManager.h"
#include "UI/SandboxCameraController.h"
#include "UI/SandboxHUD.h"
#include "Evo4XSandbox.h"

AEvo4XSandboxGameMode::AEvo4XSandboxGameMode()
{
	// Set default classes — no Blueprint needed
	PlayerControllerClass = ASandboxCameraController::StaticClass();
	HUDClass = ASandboxHUD::StaticClass();
}

void AEvo4XSandboxGameMode::BeginPlay()
{
	Super::BeginPlay();

	UWorld* World = GetWorld();
	if (!World) return;

	FActorSpawnParameters SpawnParams;
	SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	// 1. Spawn simulation controller
	SimController = World->SpawnActor<ASandboxController>(
		ASandboxController::StaticClass(),
		FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);

	// 2. Spawn map renderer
	MapRenderer = World->SpawnActor<AHexMapRenderer>(
		AHexMapRenderer::StaticClass(),
		FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);

	// 3. Spawn region overlay
	Overlay = World->SpawnActor<ARegionOverlay>(
		ARegionOverlay::StaticClass(),
		FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);

	// 4. Spawn migration arrow manager
	ArrowManager = World->SpawnActor<AMigrationArrowManager>(
		AMigrationArrowManager::StaticClass(),
		FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);

	if (!SimController || !MapRenderer || !Overlay || !ArrowManager)
	{
		UE_LOG(LogEvo4X, Error, TEXT("Failed to spawn sandbox actors!"));
		return;
	}

	// Build initial visuals
	MapRenderer->BuildMesh(SimController->GetHexGrid(), SimController->GetWorldState());
	Overlay->InitOverlays(SimController->GetHexGrid(), SimController->GetWorldState());
	ArrowManager->UpdateArrows(SimController->GetHexGrid(), SimController->GetWorldState());

	// Wire up HUD
	APlayerController* PC = World->GetFirstPlayerController();
	if (PC)
	{
		ASandboxHUD* HUD = Cast<ASandboxHUD>(PC->GetHUD());
		if (HUD)
		{
			HUD->SetWorldState(&SimController->GetWorldState());
		}
	}

	// Bind tick callback
	SimController->OnTickCompleted.AddDynamic(this, &AEvo4XSandboxGameMode::OnSimTickCompleted);
	SimController->OnRegionSelected.AddDynamic(this, &AEvo4XSandboxGameMode::OnRegionSelected);

	UE_LOG(LogEvo4X, Log, TEXT("Sandbox fully initialized. Press Space to start."));
}

void AEvo4XSandboxGameMode::OnSimTickCompleted()
{
	if (!SimController || !MapRenderer || !Overlay || !ArrowManager) return;

	const FWorldState& WS = SimController->GetWorldState();
	const FHexGrid& Grid = SimController->GetHexGrid();

	// Update visuals
	MapRenderer->UpdateColors(WS);
	Overlay->UpdateOverlays(WS);
	ArrowManager->UpdateArrows(Grid, WS);
}

void AEvo4XSandboxGameMode::OnRegionSelected(int32 RegionId)
{
	UWorld* World = GetWorld();
	if (!World) return;

	APlayerController* PC = World->GetFirstPlayerController();
	if (PC)
	{
		ASandboxHUD* HUD = Cast<ASandboxHUD>(PC->GetHUD());
		if (HUD)
		{
			HUD->SetSelectedRegion(RegionId);
		}
	}
}
