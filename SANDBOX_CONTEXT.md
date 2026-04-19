# Evo4X Sandbox — Simulation Context

> **Назначение этого файла.** Источник правды для автосима: логика тика, worldgen,
> виды, биомы, конфиг. Служит спецификацией для веб-порта (TypeScript + Vite + SVG).
> Для пошаговой игры с игроком — см. [CONTEXT.md](CONTEXT.md).

**Baseline (последнее обновление: 2026-04-19)**
- Референсная реализация: `ue5-sandbox 5.6/Source/Evo4XSandbox/SimCore/`
- Целевой стек: TypeScript + Vite + SVG (браузер)

---

## 1. Что это такое

**Evo4X Sandbox** — пространственная автосимуляция экосистемы без игрока.
Виды сами размножаются, голодают и мигрируют. Задача — наблюдать за динамикой:
кто вытесняет кого, куда текут популяции, какие регионы становятся доминирующими.

### Чего нет (в отличие от Python-игры)
- Нет игрока, нет EVO-экономики, нет трейтов, нет форков
- Нет пошаговых действий
- Нет фиксированной 3×3 сетки — карта процедурная, ~20 регионов

### Что есть
- Гексагональная карта ~500 тайлов, разбитая на ~20 регионов
- 8 видов с уникальными характеристиками, включая `Radiation` (склонность к миграции)
- Автономная миграция, управляемая давлением голода
- Биомы с ресурсными профилями и видовыми паками

---

## 2. Карта и мир

### Гексагональная сетка
- **~500 land-тайлов**, **flat-top** hex, axial координаты (Q, R)
- World-координаты: `X = HexSize × 3/2 × Q`, `Y = HexSize × √3 × (R + Q/2)`
- Тайлы делятся на `land` и `ocean` (ocean не участвует в симуляции, рендерится как border)

### Регионы
- **~20 регионов**, `MinRegionSize = 7` гексов
- Генерация: многоисточниковый BFS с приоритетом компактности (prefer tile adjacent to more same-region tiles)
- После BFS: `MergeSmallRegions` (мелкие → в крупнейшего соседа) + `FixDisconnectedRegions` (анклавы → в соседний регион)
- Регионы пронумерованы 0..N-1 после ремаппинга

### Ресурсы (3 типа)
| ID | Название | Описание |
|----|----------|----------|
| R1 | Grass | Трава |
| R2 | Leaves | Листья |
| R3 | Roots | Корни |

### Богатство региона
`BaseR_i = BiomeRatio_i × RichnessPerHex × HexCount`

Т.е. богатство масштабируется с размером региона. `RichnessPerHex = 10` (дефолт).

---

## 3. Биомы

9 биомов, каждый задаёт ресурсный профиль (R1/R2/R3 в %) и список паков видов.

| ID | Название | R1% | R2% | R3% | Цвет |
|----|----------|-----|-----|-----|------|
| `grassland` | Grassland | 100 | 0 | 0 | Vivid lime |
| `open_savanna` | Open Savanna | 67 | 33 | 0 | Golden yellow |
| `woodland` | Woodland | 33 | 67 | 0 | Deep forest green |
| `dense_grove` | Dense Grove | 20 | 80 | 0 | Dark jungle green |
| `root_patch` | Root Patch | 67 | 0 | 33 | Warm amber |
| `rootland` | Rootland | 33 | 0 | 67 | Burnt orange-brown |
| `diverse_savanna` | Diverse Savanna | 50 | 30 | 20 | Yellow-green |
| `root_mosaic` | Root Mosaic | 40 | 20 | 40 | Deep ochre |
| `leaf_mosaic` | Leaf Mosaic | 30 | 50 | 20 | Teal-green |

### Паки видов по биому
При генерации для каждого региона случайно выбирается один пак (пустой пак = регион без видов).

| Биом | Паки |
|------|------|
| `grassland` | `[]`, `[gazelle]`, `[zebra]`, `[buffalo]` |
| `open_savanna` | `[giraffe]`, `[giraffe, gazelle]`, `[giraffe, zebra]`, `[giraffe, buffalo]`, `[giraffe, warthog]`, `[giraffe, impala]`, `[impala]` |
| `woodland` | `[giraffe]`, `[giraffe, buffalo]`, `[giraffe, impala]` |
| `dense_grove` | `[giraffe]`, `[elephant]` |
| `root_patch` | `[mole_rat]`, `[mole_rat, gazelle]`, `[mole_rat, zebra]`, `[mole_rat, buffalo]`, `[mole_rat, giraffe]`, `[warthog]`, `[warthog, gazelle]`, `[warthog, impala]`, `[warthog, buffalo]`, `[warthog, elephant]` |
| `rootland` | `[mole_rat]`, `[mole_rat, gazelle]`, `[mole_rat, zebra]`, `[mole_rat, buffalo]`, `[mole_rat, giraffe]`, `[mole_rat, elephant]`, `[warthog]` |
| `diverse_savanna` | `[giraffe, mole_rat, gazelle]`, `[giraffe, mole_rat, impala]`, `[giraffe, mole_rat, buffalo]`, `[giraffe, warthog, impala]`, `[giraffe, warthog, buffalo]`, `[giraffe, mole_rat, zebra]` |
| `root_mosaic` | `[giraffe, mole_rat, gazelle]`, `[giraffe, mole_rat, impala]`, `[giraffe, mole_rat, buffalo]`, `[giraffe, warthog, impala]`, `[giraffe, warthog, buffalo]`, `[giraffe, mole_rat, zebra]` |
| `leaf_mosaic` | `[giraffe, mole_rat, gazelle]`, `[giraffe, mole_rat, impala]`, `[giraffe, mole_rat, buffalo]`, `[giraffe, warthog, buffalo]` |

---

## 4. Виды

8 видов. Статы фиксированы (нет трейтов, нет апгрейдов).

| ID | Название | r | bR1 | bR2 | bR3 | Radiation% |
|----|----------|---|-----|-----|-----|------------|
| `gazelle` | Gazelle | 1.00 | 1.0 | 0.0 | 0.0 | 8 |
| `impala` | Impala | 0.95 | 1.2 | 0.6 | 0.0 | 6 |
| `zebra` | Zebra | 0.75 | 1.1 | 0.0 | 0.0 | 5 |
| `buffalo` | Buffalo | 0.65 | 2.0 | 0.0 | 0.0 | 3 |
| `giraffe` | Giraffe | 0.40 | 0.5 | 3.0 | 0.0 | 4 |
| `elephant` | Elephant | 0.12 | 2.0 | 3.0 | 0.0 | 2 |
| `warthog` | Warthog | 1.20 | 0.5 | 0.0 | 2.0 | 6 |
| `mole_rat` | Mole Rat | 1.80 | 0.0 | 0.0 | 1.0 | 5 |

- **r** — прирост за тик: `births = pop × r`
- **bR1/bR2/bR3** — веса потребления ресурсов при allocation
- **Radiation** — % шанс что одна голодающая пара мигрирует (Bernoulli per pair)

### Стартовая популяция
`StartPop = Region.HexCount × StartPopDensity` (дефолт `StartPopDensity = 2`)

---

## 5. Логика тика (двухфазовая)

```
SimulateTick(World):
  ── PHASE 1: каждый регион независимо ──
  for each Region:
    1. ApplyActiveEffects   → Region.EffR1/R2/R3
    2. ComputeBirths        → Breakdown[sp].PopPreBirth
    3. AllocateResources    → Breakdown[sp].FoodAllocated
    4. ComputeMigration     → World.PendingMigrants, MigrationEdgesThisTick
    5. CommitStarvation     → Region.Populations[sp]

  ── PHASE 2: глобальные операции ──
  6. ApplyIncomingMigrants  → добавляет PendingMigrants в популяции
  7. ExtinctionCleanup      → удаляет sp где pop <= ExtinctionEps
  8. TickDownEffects        → TTL-- для каждого ActiveEffect
  9. UpdateMetrics          → тренды, история, доминант по регионам
  10. World.CurrentTick++
```

### Шаг 1 — ApplyActiveEffects
```
Region.EffRi = Region.BaseRi
for each Effect in Region.ActiveEffects:
    EffRi *= Effect.MultRi
```

### Шаг 2 — ComputeBirths
```
PopPreBirth[sp] = Population[sp] × (1 + r[sp])
```

### Шаг 3 — AllocateResources
Для каждого ресурса Ri:
```
w[sp] = PopPreBirth[sp] × b[sp][Ri]
FoodAllocated[sp] += EffRi × w[sp] / Σw
```
Пропорциональное деление по весу потребления.

### Шаг 4 — ComputeMigration (ключевая механика)
```
StarvingNewborns[sp] = min(births, max(0, PopPreBirth - FoodAllocated))
StarvingPairs = floor(StarvingNewborns / 2)

for each pair in 0..StarvingPairs-1:
    roll = Rand01(seed, "migrate|tick|regionId|sp|pairIdx")
    if roll < Radiation[sp] / 100:
        NumExpeditions++

OutgoingMigrants[sp] = NumExpeditions × 2

# Распределение по соседям:
Base = NumExpeditions / len(Neighbors)   # каждому соседу
Rem  = NumExpeditions % len(Neighbors)   # остаток — детерминированно (shuffle)

→ добавляет в World.PendingMigrants[neighbor][sp]
→ добавляет в World.MigrationEdgesThisTick (для рендера стрелок)
```

### Шаг 5 — CommitStarvation
```
EffectiveNPre[sp] = PopPreBirth[sp] - OutgoingMigrants[sp]
Survivors[sp] = min(EffectiveNPre, FoodAllocated[sp])
Region.Populations[sp] = max(0, Survivors)
```
**Важно:** мигранты вычитаются из n_pre ДО сравнения с едой.

### Шаг 6 — ApplyIncomingMigrants
Мигранты из PendingMigrants просто прибавляются к популяции региона-получателя.
Они не голодают в тике прибытия — только со следующего тика.

### Шаг 7 — ExtinctionCleanup
```
if Population[sp] <= ExtinctionEps (=1.0):
    удалить sp из Populations и TickBreakdown
```

### Шаг 8 — TickDownEffects
```
for each Effect: TTL--
удалить если TTL <= 0
```

### Шаг 9 — UpdateMetrics
- `Region.TotalBiomass = Σ pop`
- `Region.DominantSpecies` — вид с макс популяцией
- `PopHistory[sp]` — ring buffer последних `TrendHistoryLen=50` тиков
- Тренд: сравнение с `TrendWindowShort=3` тиков назад, порог `±5%` → Up/Down/Flat
- Глобальные агрегаты: `TotalWorldBiomass`, `AliveSpeciesCount`, `FastestGrowing/Declining`

---

## 6. Генерация мира (WorldGen pipeline)

```
Generate(Config, OutGrid, OutWorld):
  1. GenerateContinent       → органический blob ~TotalHexes тайлов
  2. GenerateRegions         → многоисточниковый BFS с компактностью
  3. MergeSmallRegions       → мелкие регионы → в крупнейшего соседа
  4. FixDisconnectedRegions  → анклавы → в соседний регион
  5. Build Region structs + remap IDs → 0..N-1
  6. BuildAdjacency          → Region.Neighbors[] через hex-соседство
  7. AssignBiomes            → uniform random из 9 биомов per region
  8. ComputeBaseResources    → BaseRi = Ratio × RichnessPerHex × HexCount
  9. SeedSpecies             → random pack из биома, StartPop = HexCount × StartPopDensity
```

### Шаг 1 — GenerateContinent (noise-based expansion)
Не чистый BFS — вероятностное расширение с убыванием по расстоянию:
```
MaxRadius = ceil(sqrt(TargetLandCount) × 1.2)
Frontier = [center]

while LandCount < TargetLandCount:
    pick random tile from Frontier   (key: "continent|pick|{iter}")
    pick random neighbor             (key: "continent|nb|{iter}")
    dist = hex_distance(center, neighbor)

    if dist > MaxRadius + 2: skip
    AcceptChance = clamp(1 - dist / MaxRadius, 0.1, 1.0)
    roll = Rand01(seed, "continent|accept|{iter}")
    if roll > AcceptChance: skip     # отклонено — дальние клетки реже принимаются

    mark neighbor as land, add to Frontier
    if all neighbors of current tile are land: remove from Frontier
```
**Результат:** органический blob с плотным центром и изрезанными краями.
Форма воспроизводима при том же `WorldSeed`.

### Шаг 2 — GenerateRegions (compactness BFS)
```
# 1. Выбрать NumRegions seed-точек с минимальным расстоянием между ними
MinSeedDist = max(2, floor(sqrt(LandCount / NumRegions) × 0.5))
SeedPoints = spread_seeds(shuffled_land, NumRegions, MinSeedDist)

# 2. Многоисточниковый BFS — каждый регион расширяется по очереди за раунд
for each round:
    for each region:
        candidates = unassigned neighbors of current frontier
        best_score = max tiles adjacent to same region   # компактность
        claim tile with best_score (детерминированно)
```
Если seed-точек не хватает (слишком плотная карта) — добираем без ограничения дистанции.

### Шаг 3 — MergeSmallRegions
Итеративно: найти регион с `size < MinRegionSize` → слить в крупнейшего соседа → повторить до стабильности.

### Шаг 4 — FixDisconnectedRegions
BFS flood fill внутри каждого региона. Тайлы, не достижимые от главного компонента → передать ближайшему соседнему региону. Повторять до стабильности.

### Шаг 6 — BuildAdjacency
Для каждого land-тайла: если сосед принадлежит другому региону → добавить пару (RId, NbRegion) в `Neighbors[]` обоих регионов. Дупликаты исключаются через packed uint64 set.

### Шаг 7 — AssignBiomes
Каждому региону случайно назначается один из 9 биомов: `ChooseIndex(seed, "biome_assign|{regionId}", 9)`. Никаких ограничений на соседство.

### Шаг 9 — SeedSpecies
```
for each region:
    pack = random non-empty pack из биома (key: "species_pack|{regionId}")
    for each species in pack:
        Population[species] = HexCount × StartPopDensity
```

### RNG
Стейтлесс SHA256: `Rand01(seed, key) = sha256("{seed}|{key}")[:8] as uint32 / 2^32`
Идентичный алгоритм с Python-движком. Все случайные выборы используют осмысленный строковый ключ.

---

## 7. Конфиг (SimConfig)

| Параметр | Дефолт | Описание |
|---|---|---|
| `TotalHexes` | 500 | Примерное число land-тайлов |
| `NumRegions` | 20 | Целевое число регионов |
| `MinRegionSize` | 7 | Минимум гексов в регионе |
| `RichnessPerHex` | 10 | Ресурсный бюджет на гекс |
| `StartPopDensity` | 2 | Стартовая поп на гекс для каждого вида в паке |
| `ExtinctionEps` | 1.0 | Порог вымирания |
| `bEnableEvents` | false | Флуктуационные события (архитектура есть, пока выкл) |
| `TrendHistoryLen` | 50 | Тиков истории на регион |
| `TrendWindowShort` | 3 | Окно для тренд-стрелки |
| `TrendThreshold` | 0.05 | ±5% = Flat |
| `WorldSeed` | "42" | Сид для всего worldgen |

---

## 8. Целевая архитектура (TypeScript + Vite + SVG)

```
src/
  sim/
    types.ts          — интерфейсы: Region, Species, WorldState, MigrationEdge
    rng.ts            — Rand01, ChooseIndex (SHA256, совместим с UE5/Python)
    biomes.ts         — 9 биомов с ратио и паками
    species.ts        — 8 видов со всеми статами
    worldgen.ts       — Generate pipeline
    simulation.ts     — SimulateTick двухфазовый
  render/
    hexmap.ts         — SVG hex grid, клик → выбор региона
    arrows.ts         — миграционные стрелки (SVG line + marker)
    inspector.ts      — боковая панель региона
    hud.ts            — глобальные метрики
  main.ts
index.html
```

### Ключевые решения по рендеру
- Регионы = SVG `<polygon>` (union гексов через path merge или просто отдельные hex с одинаковым fill)
- Клик = `onclick` на SVG-элементе региона
- Стрелки = `<line>` + `<marker defs>` для наконечника, толщина = f(кол-во мигрантов)
- Хитмап популяции = `fill` цвет + opacity
- Тренд = маленький `▲▼` текст или иконка поверх региона

---

## 9. Отличия от Python-движка (Evo4X)

| Аспект | Python (Evo4X) | Sandbox |
|---|---|---|
| Карта | 3×3 = 9 клеток | ~500 гексов, ~20 регионов |
| Worldgen | Детерминированные слоты, жёсткие инварианты | Процедурный BFS, random биомы |
| Богатство | Фиксировано: poor=30, normal=60, rich=90 | `RichnessPerHex × HexCount` |
| Миграция | **Игрок**, явный action dict, до 50% популяции | **Автономная**, Bernoulli per starving pair × Radiation |
| Radiation | Нет | Стат каждого вида (2–8%) |
| Starvation | `survivors = min(n_pre, food)` | `survivors = min(n_pre - outgoing, food)` |
| Порядок тика | effects → births → allocation → commit | births → allocation → migration → commit → apply (двухфазовый) |
| Игрок | Есть: EVO, трейты, форк | Нет |
| События | Включены (fluctuation, TTL 3–6) | Архитектура есть, `bEnableEvents=false` |

---

## 10. Открытые вопросы / следующие шаги

1. **Веб-порт**: реализовать `sim/` модули на TypeScript, верифицировать детерминизм через тест с известным сидом
2. **Совместимость RNG**: SHA256 в браузере — использовать `crypto.subtle.digest` или js-реализацию
3. **Рендер регионов**: решить polygon-union vs hex-by-hex (влияет на читаемость границ)
4. **События**: когда включать `bEnableEvents` — после того как базовая симуляция стабилизирована
5. **Баланс**: после порта проверить что динамика аналогична UE5-версии (тот же сид → та же картина через 100 тиков)
