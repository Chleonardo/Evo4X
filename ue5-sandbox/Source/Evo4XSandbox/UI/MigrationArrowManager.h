#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/LineBatchComponent.h"
#include "SimCore/WorldState.h"
#include "MapGen/HexGrid.h"
#include "MigrationArrowManager.generated.h"

/**
 * Renders migration arrows between regions using debug lines.
 * Arrow thickness/brightness = number of migrants.
 * Updated each tick.
 */
UCLASS()
class EVO4XSANDBOX_API AMigrationArrowManager : public AActor
{
	GENERATED_BODY()

public:
	AMigrationArrowManager();

	/** Update migration arrows from current world state. */
	UFUNCTION(BlueprintCallable, Category = "Migration")
	void UpdateArrows(const FHexGrid& Grid, const FWorldState& World);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Migration")
	float ArrowZHeight = 10.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Migration")
	float MaxArrowThickness = 5.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Migration")
	float MinArrowThickness = 1.f;

protected:
	virtual void Tick(float DeltaTime) override;

private:
	/** Cached arrow data for rendering. */
	struct FArrowData
	{
		FVector Start;
		FVector End;
		FColor Color;
		float Thickness;
	};

	TArray<FArrowData> CachedArrows;
	bool bNeedsRedraw = false;

	/** Compute center of a region in world space. */
	static FVector2D GetRegionCenter(const FHexGrid& Grid, const FRegion& Region);
};
