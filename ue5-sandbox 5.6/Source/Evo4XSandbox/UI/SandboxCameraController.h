#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "SandboxCameraController.generated.h"

class ASandboxController;
class AHexMapRenderer;

/**
 * Top-down orthographic camera with pan (WASD/drag) and zoom (scroll).
 * Click to select region.
 */
UCLASS()
class EVO4XSANDBOX_API ASandboxCameraController : public APlayerController
{
	GENERATED_BODY()

public:
	ASandboxCameraController();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	float PanSpeed = 500.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	float ZoomSpeed = 200.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	float MinZoom = 1000.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	float MaxZoom = 15000.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	float InitialZoom = 6000.f;

protected:
	virtual void BeginPlay() override;
	virtual void SetupInputComponent() override;
	virtual void Tick(float DeltaTime) override;

private:
	UPROPERTY()
	AActor* CameraActor;

	float CurrentZoom;

	// Input state
	float PanX = 0.f;
	float PanY = 0.f;
	bool bDragging = false;
	FVector2D DragStart;

	void OnPanX(float Value);
	void OnPanY(float Value);
	void OnZoomIn();
	void OnZoomOut();
	void OnClick();
	void OnTogglePause();
	void OnSetSpeedSlow();
	void OnSetSpeedFast();
	void OnSetSpeedVeryFast();
	void OnStepTick();
	void OnCycleOverlay();
};
