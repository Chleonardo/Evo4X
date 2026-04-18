#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "Evo4XSandboxGameMode.generated.h"

class ASandboxController;
class AHexMapRenderer;
class ARegionOverlay;
class AMigrationArrowManager;

/**
 * Game mode that auto-spawns all sandbox actors on BeginPlay.
 * No manual actor placement needed — just Play.
 */
UCLASS()
class EVO4XSANDBOX_API AEvo4XSandboxGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	AEvo4XSandboxGameMode();

protected:
	virtual void BeginPlay() override;

private:
	UPROPERTY()
	ASandboxController* SimController = nullptr;

	UPROPERTY()
	AHexMapRenderer* MapRenderer = nullptr;

	UPROPERTY()
	ARegionOverlay* Overlay = nullptr;

	UPROPERTY()
	AMigrationArrowManager* ArrowManager = nullptr;

	/** Called after each simulation tick — refreshes all visuals. */
	UFUNCTION()
	void OnSimTickCompleted();

	/** Called when a region is selected — updates HUD. */
	UFUNCTION()
	void OnRegionSelected(int32 RegionId);
};
