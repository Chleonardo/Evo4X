# Evo4X Sandbox — UE5 Spatial Simulation Prototype

Observation-only ecosystem sandbox: watch NPC species compete, grow, and migrate
across a hex-based continent with 20 biome regions.

## Quick Start

1. Open `Evo4XSandbox.uproject` in Unreal Engine 5.4+
2. Compile (C++ project)
3. In the editor, the world auto-generates on Play
4. Use keyboard controls to observe

## Controls

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| 1 | Slow speed (1 tick / 5 seconds) |
| 2 | Fast speed (1 tick / 1 second) |
| T | Step one tick (while paused) |
| O | Cycle overlay mode |
| WASD | Pan camera |
| Scroll | Zoom in/out |
| Click | Select region (open inspector) |

## Overlay Modes (cycle with O)

1. **Biomes** — color = biome type
2. **Dominant Species** — color = leading species
3. **Total Biomass** — brightness = population density
4. **Resource R1** (Grass) — intensity map
5. **Resource R2** (Leaves) — intensity map
6. **Resource R3** (Roots) — intensity map
7. **Resource Total** — total richness

## Where to Change Things

### Species (stats, radiation, colors)
`Source/Evo4XSandbox/SimCore/SpeciesData.cpp` — `FSpeciesRegistry::Initialize()`

Each species has:
- `R` — growth rate per tick
- `BR1, BR2, BR3` — resource affinity weights
- `Radiation` — % chance each starving pair migrates (0-100)
- `Color` — map overlay color

### Biome Settings (ratios, colors, packs)
`Source/Evo4XSandbox/SimCore/BiomeData.cpp` — `FBiomeRegistry::Initialize()`

Each biome has:
- `RatioR1, R2, R3` — resource distribution (normalized)
- `Color` — map tile color
- `Packs` — which species can spawn together

### Simulation Config
`Source/Evo4XSandbox/SimCore/SimConfig.h` — `FSimConfig` struct

Key parameters:
- `RichnessPerHex = 10` — resources per hex tile
- `StartPopDensity = 2` — starting pop per species = hex_count * this
- `ExtinctionEps = 1.0` — species dies when pop falls to this
- `TickSpeedSlow = 5.0` — seconds per tick in slow mode
- `TickSpeedFast = 1.0` — seconds per tick in fast mode
- `TotalHexes = 500` — target continent size
- `NumRegions = 20` — number of regions

### World Seed
Set on the `ASandboxController` actor in the level (property `WorldSeed`).
Default: `"42"`. Same seed = same map + same simulation.

### Events
`SimConfig.h` → `bEnableEvents = false` (default off).
Architecture is ready: `FActiveEffect` in `Region.h` supports resource multipliers with TTL.
To enable, set to `true` and implement event rolling in `SimulationCore::SimulateTick`.

## Architecture

```
Source/Evo4XSandbox/
├── SimCore/           # Pure C++ simulation (no UE rendering deps)
│   ├── EvoRng         # Deterministic SHA-256 RNG
│   ├── BiomeData      # 9 biome archetypes
│   ├── SpeciesData    # 8 species with stats + radiation
│   ├── SimConfig      # All tunable parameters
│   ├── Region         # Region state, effects, breakdown
│   ├── WorldState     # Full world + metrics
│   └── SimulationCore # Tick logic: births, allocation, starvation, migration
├── MapGen/            # World generation
│   ├── HexGrid        # Hex coordinate system + continent flood fill
│   └── WorldGenerator # Regions, biomes, species seeding
└── UI/                # UE presentation layer
    ├── SandboxController       # Game flow: init, tick loop, speed
    ├── SandboxCameraController # Top-down camera: pan, zoom, click
    ├── HexMapRenderer          # Procedural mesh hex map + overlays
    ├── RegionOverlay           # Text labels on regions
    ├── MigrationArrowManager   # Migration arrows between regions
    └── SandboxHUD              # Canvas HUD: panels, inspector, graph
```

## Simulation Tick Order

1. Apply active effects (if enabled)
2. Births: `n_pre = pop * (1 + r)`
3. Resource allocation (proportional by weight `n_pre * b[R]`)
4. Identify starving newborns
5. Compute outgoing expeditions (Bernoulli per pair, `radiation%` chance)
6. Local starvation commit
7. Apply incoming migrants (two-phase: all regions finish phase 1 first)
8. Extinction cleanup
9. Update metrics and trends
10. Advance tick counter
