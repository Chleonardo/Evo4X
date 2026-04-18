#include "SandboxCameraController.h"
#include "SandboxController.h"
#include "HexMapRenderer.h"
#include "Camera/CameraActor.h"
#include "Camera/CameraComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/World.h"
#include "Evo4XSandbox.h"

ASandboxCameraController::ASandboxCameraController()
{
	bShowMouseCursor = true;
	bEnableClickEvents = true;
	CurrentZoom = 6000.f;
}

void ASandboxCameraController::BeginPlay()
{
	Super::BeginPlay();

	CurrentZoom = InitialZoom;

	// Spawn a camera actor for top-down view
	FActorSpawnParameters SpawnParams;
	SpawnParams.Owner = this;
	CameraActor = GetWorld()->SpawnActor<ACameraActor>(
		FVector(0.f, 0.f, CurrentZoom),
		FRotator(-90.f, 0.f, 0.f),
		SpawnParams
	);

	if (CameraActor)
	{
		ACameraActor* Cam = Cast<ACameraActor>(CameraActor);
		if (Cam && Cam->GetCameraComponent())
		{
			Cam->GetCameraComponent()->ProjectionMode = ECameraProjectionMode::Orthographic;
			Cam->GetCameraComponent()->OrthoWidth = CurrentZoom * 2.f;
		}
		SetViewTarget(CameraActor);
	}
}

void ASandboxCameraController::SetupInputComponent()
{
	Super::SetupInputComponent();

	if (!InputComponent) return;

	// WASD pan
	InputComponent->BindAxis("MoveRight", this, &ASandboxCameraController::OnPanX);
	InputComponent->BindAxis("MoveForward", this, &ASandboxCameraController::OnPanY);

	// Zoom
	InputComponent->BindAction("ZoomIn",  IE_Pressed, this, &ASandboxCameraController::OnZoomIn);
	InputComponent->BindAction("ZoomOut", IE_Pressed, this, &ASandboxCameraController::OnZoomOut);
	InputComponent->BindKey(EKeys::Three, IE_Pressed, this, &ASandboxCameraController::OnSetSpeedVeryFast);

	// Click
	InputComponent->BindAction("Select", IE_Pressed, this, &ASandboxCameraController::OnClick);

	// Simulation controls
	InputComponent->BindAction("TogglePause", IE_Pressed, this, &ASandboxCameraController::OnTogglePause);
	InputComponent->BindAction("SpeedSlow", IE_Pressed, this, &ASandboxCameraController::OnSetSpeedSlow);
	InputComponent->BindAction("SpeedFast", IE_Pressed, this, &ASandboxCameraController::OnSetSpeedFast);
	InputComponent->BindAction("StepTick", IE_Pressed, this, &ASandboxCameraController::OnStepTick);
	InputComponent->BindAction("CycleOverlay", IE_Pressed, this, &ASandboxCameraController::OnCycleOverlay);
}

void ASandboxCameraController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!CameraActor) return;

	// Apply pan
	if (PanX != 0.f || PanY != 0.f)
	{
		const float Scale = CurrentZoom / InitialZoom; // pan faster when zoomed out
		FVector Loc = CameraActor->GetActorLocation();
		Loc.X += PanY * PanSpeed * Scale * DeltaTime;
		Loc.Y += PanX * PanSpeed * Scale * DeltaTime;
		CameraActor->SetActorLocation(Loc);
	}
}

void ASandboxCameraController::OnPanX(float Value) { PanX = Value; }
void ASandboxCameraController::OnPanY(float Value) { PanY = Value; }

void ASandboxCameraController::OnZoomIn()
{
	CurrentZoom = FMath::Max(MinZoom, CurrentZoom - ZoomSpeed);
	if (CameraActor)
	{
		FVector Loc = CameraActor->GetActorLocation();
		Loc.Z = CurrentZoom;
		CameraActor->SetActorLocation(Loc);

		ACameraActor* Cam = Cast<ACameraActor>(CameraActor);
		if (Cam && Cam->GetCameraComponent())
		{
			Cam->GetCameraComponent()->OrthoWidth = CurrentZoom * 2.f;
		}
	}
}

void ASandboxCameraController::OnZoomOut()
{
	CurrentZoom = FMath::Min(MaxZoom, CurrentZoom + ZoomSpeed);
	if (CameraActor)
	{
		FVector Loc = CameraActor->GetActorLocation();
		Loc.Z = CurrentZoom;
		CameraActor->SetActorLocation(Loc);

		ACameraActor* Cam = Cast<ACameraActor>(CameraActor);
		if (Cam && Cam->GetCameraComponent())
		{
			Cam->GetCameraComponent()->OrthoWidth = CurrentZoom * 2.f;
		}
	}
}

void ASandboxCameraController::OnClick()
{
	// For orthographic top-down camera we don't need a mesh hit —
	// just deproject the mouse to world space and use XY directly.
	FVector WorldLocation, WorldDirection;
	if (!DeprojectMousePositionToWorld(WorldLocation, WorldDirection)) return;

	// Intersect ray with Z=0 plane
	const float DirZ = WorldDirection.Z;
	if (FMath::Abs(DirZ) < KINDA_SMALL_NUMBER) return;
	const float T = -WorldLocation.Z / DirZ;
	const FVector HitPoint = WorldLocation + WorldDirection * T;

	AHexMapRenderer* MapRenderer = Cast<AHexMapRenderer>(
		UGameplayStatics::GetActorOfClass(GetWorld(), AHexMapRenderer::StaticClass()));
	ASandboxController* Sim = Cast<ASandboxController>(
		UGameplayStatics::GetActorOfClass(GetWorld(), ASandboxController::StaticClass()));

	if (MapRenderer && Sim)
	{
		const int32 RegionId = MapRenderer->WorldPosToRegion(HitPoint);
		if (RegionId >= 0)
		{
			Sim->SelectRegion(RegionId);
		}
	}
}

void ASandboxCameraController::OnTogglePause()
{
	ASandboxController* Sim = Cast<ASandboxController>(
		UGameplayStatics::GetActorOfClass(GetWorld(), ASandboxController::StaticClass()));
	if (Sim) Sim->TogglePause();
}

void ASandboxCameraController::OnSetSpeedSlow()
{
	ASandboxController* Sim = Cast<ASandboxController>(
		UGameplayStatics::GetActorOfClass(GetWorld(), ASandboxController::StaticClass()));
	if (Sim) Sim->SetSpeed(1);
}

void ASandboxCameraController::OnSetSpeedFast()
{
	ASandboxController* Sim = Cast<ASandboxController>(
		UGameplayStatics::GetActorOfClass(GetWorld(), ASandboxController::StaticClass()));
	if (Sim) Sim->SetSpeed(2);
}

void ASandboxCameraController::OnSetSpeedVeryFast()
{
	ASandboxController* Sim = Cast<ASandboxController>(
		UGameplayStatics::GetActorOfClass(GetWorld(), ASandboxController::StaticClass()));
	if (Sim) Sim->SetSpeed(3);
}

void ASandboxCameraController::OnStepTick()
{
	ASandboxController* Sim = Cast<ASandboxController>(
		UGameplayStatics::GetActorOfClass(GetWorld(), ASandboxController::StaticClass()));
	if (Sim)
	{
		Sim->SetSpeed(0); // pause
		Sim->StepOneTick();
	}
}

void ASandboxCameraController::OnCycleOverlay()
{
	AHexMapRenderer* MapRenderer = Cast<AHexMapRenderer>(
		UGameplayStatics::GetActorOfClass(GetWorld(), AHexMapRenderer::StaticClass()));
	ASandboxController* Sim = Cast<ASandboxController>(
		UGameplayStatics::GetActorOfClass(GetWorld(), ASandboxController::StaticClass()));

	if (MapRenderer && Sim)
	{
		int32 Current = static_cast<int32>(MapRenderer->CurrentOverlay);
		Current = (Current + 1) % 7; // 7 overlay modes
		MapRenderer->SetOverlay(static_cast<EMapOverlay>(Current), Sim->GetWorldState());
	}
}
