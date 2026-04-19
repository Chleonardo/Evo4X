# Evo4X Sandbox — Simulation Context

> **Назначение этого файла.** Источник правды для автосима: логика тика, worldgen,
> виды, биомы, конфиг. Служит спецификацией для веб-порта (TypeScript + Vite + SVG).
> Для пошаговой игры с игроком — см. [CONTEXT.md](CONTEXT.md).

**Baseline (последнее обновление: 2026-04-20)**
- Целевой стек: TypeScript + Vite + SVG (браузер)
- Директория: `web-sandbox/`
- Dev: `cd web-sandbox && npm run dev` → `localhost:5173`
- Food lab: `localhost:5173/food-lab.html`

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
- Save/load (JSON: seed + tick + state)
- Food lab — изолированный харнесс для тестирования food mechanics

---

## 2. Карта и мир

### Гексагональная сетка
- **~500 land-тайлов**, **flat-top** hex, axial координаты (Q, R)
- World-координаты: `X = HexSize × 3/2 × Q`, `Y = HexSize × √3 × (R + Q/2)`
- Тайлы делятся на `land` и `ocean` (ocean = синий фон `#1e6091`, не участвует в симуляции)

### Регионы
- **~20 регионов**, `MinRegionSize = 7` гексов
- Генерация: **Voronoi-присвоение** (заменило BFS-рост) — каждый тайл назначается ближайшему seed-региону по hex-расстоянию. Даёт компактные, незакрученные регионы.
- После Voronoi: мерж маленьких регионов в крупнейшего соседа
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

| ID | Название | R1% | R2% | R3% |
|----|----------|-----|-----|-----|
| `grassland` | Grassland | 100 | 0 | 0 |
| `open_savanna` | Open Savanna | 67 | 33 | 0 |
| `woodland` | Woodland | 33 | 67 | 0 |
| `dense_grove` | Dense Grove | 20 | 80 | 0 |
| `root_patch` | Root Patch | 67 | 0 | 33 |
| `rootland` | Rootland | 33 | 0 | 67 |
| `diverse_savanna` | Diverse Savanna | 50 | 30 | 20 |
| `root_mosaic` | Root Mosaic | 40 | 20 | 40 |
| `leaf_mosaic` | Leaf Mosaic | 30 | 50 | 20 |

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
| `root_mosaic` | (same as diverse_savanna) |
| `leaf_mosaic` | `[giraffe, mole_rat, gazelle]`, `[giraffe, mole_rat, impala]`, `[giraffe, mole_rat, buffalo]`, `[giraffe, warthog, buffalo]` |

---

## 4. Виды

8 видов. Статы в `web-sandbox/src/sim/species.ts`.

### Основные статы (используются в main sandbox simulation)
| ID | r | bR1 | bR2 | bR3 | Radiation% | size | consumption |
|----|---|-----|-----|-----|------------|------|-------------|
| `gazelle` | 1.00 | 1.0 | 0.0 | 0.0 | 8 | 1.0 | 1.0 |
| `impala` | 0.95 | 1.2 | 0.6 | 0.0 | 6 | 2.0 | 2.0 |
| `zebra` | 0.75 | 1.1 | 0.0 | 0.0 | 5 | 3.0 | 3.0 |
| `buffalo` | 0.65 | 2.0 | 0.0 | 0.0 | 3 | 6.5 | 6.5 |
| `giraffe` | 0.40 | 0.5 | 3.0 | 0.0 | 4 | 7.0 | 7.0 |
| `elephant` | 0.12 | 2.0 | 3.0 | 0.0 | 2 | 10.0 | 10.0 |
| `warthog` | 1.20 | 0.5 | 0.0 | 2.0 | 6 | 1.8 | 1.8 |
| `mole_rat` | 1.80 | 0.0 | 0.0 | 1.0 | 5 | 0.2 | 0.2 |

> **Важно:** поля `size` и `consumption` добавлены в схему, но **не применяются в основном
> simulation loop** (main sandbox использует формулу без consumption). Они используются
> только в food-lab харнессе (`web-sandbox/src/lab/labsim.ts`).

- **r** — прирост за тик: `births = pop × r`
- **bR1/bR2/bR3** — веса потребления ресурсов при allocation
- **Radiation** — % шанс что одна голодающая пара мигрирует (Bernoulli per pair)
- **size / consumption** — размер тела / прожорливость (= size, v1 правило)

### Стартовая популяция
`StartPop = Region.HexCount × StartPopDensity` (дефолт `StartPopDensity = 2`)

---

## 5. Логика тика (двухфазовая, основной sandbox)

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
  10. SmoothMigrationEdges  → exponential decay для стрелок
  11. World.CurrentTick++
```

### Шаг 3 — AllocateResources (формула без consumption)
```
w[sp] = PopPreBirth[sp] × b[sp][Ri]
FoodAllocated[sp] += EffRi × w[sp] / Σw
```

### Шаг 4 — ComputeMigration
```
StarvingNewborns[sp] = min(births, max(0, PopPreBirth - FoodAllocated))
StarvingPairs = floor(StarvingNewborns / 2)

for each pair → Bernoulli roll < Radiation[sp]/100 → NumExpeditions++

OutgoingMigrants[sp] = NumExpeditions × 2
Base = NumExpeditions / len(Neighbors), Rem = NumExpeditions % len(Neighbors)
→ PendingMigrants[neighbor][sp] += count
→ MigrationEdgesThisTick append
```

### Шаг 5 — CommitStarvation
```
EffectiveNPre[sp] = PopPreBirth[sp] - OutgoingMigrants[sp]
Survivors[sp] = min(EffectiveNPre, FoodAllocated[sp])
```

### Шаг 10 — SmoothMigrationEdges
```
DECAY = 0.92, THRESHOLD = 0.1
for each edge in migrationSmoothed: count *= DECAY; if count < 0.1 → delete
for each edge in this tick: migrationSmoothed[key].count += edge.count
```
Ключ: `"fromRegion->toRegion:speciesId"`. Используется только для рендера стрелок.

---

## 6. Food Lab (isolated harness)

Отдельная страница `food-lab.html` для отладки food mechanics.
Не трогает основной simulation loop.

### Особенности
- Нет миграции — изолированная 1-клетка
- Тест-кейсы берутся из `biome.packs` выбранного биома
- Начальная популяция: 50 особей каждого вида в паке
- Ресурсный пул: 200 (= 20 гексов × richnessPerHex=10)

### Новая формула (food lab v1)
**Allocation:**
```
w[sp] = PopPreBirth[sp] × LAB_CONSUMPTION[sp] × b[sp][Ri]
FoodAllocated[sp] += EffRi × w[sp] / Σw
```
**Survival:**
```
Survivors[sp] = min(PopPreBirth[sp], FoodAllocated[sp] / LAB_CONSUMPTION[sp])
```

### LAB_CONSUMPTION (lab-only, сжатый диапазон)
| Вид | Lab consumption | (spec value) |
|-----|-----------------|--------------|
| gazelle | 1.0 | 1.0 |
| impala | 1.3 | 2.0 |
| zebra | 1.6 | 3.0 |
| buffalo | 2.2 | 6.5 |
| giraffe | 2.5 | 7.0 |
| elephant | 3.5 | 10.0 |
| warthog | 1.2 | 1.8 |
| mole_rat | 0.5 | 0.2 |

Диапазон сжат 0.5–3.5 (вместо 0.2–10) чтобы нишевое разделение по bRi
доминировало над raw-размером при межвидовой конкуренции.

### Файлы
- `web-sandbox/src/lab/labsim.ts` — логика тика с consumption
- `web-sandbox/src/food-lab.ts` — UI, биом-селектор, рендер графиков
- `web-sandbox/food-lab.html` — standalone Vite page

---

## 7. Рендер

### Слои SVG (порядок)
1. `fillLayer` — hex-полигоны с цветом биома, клик → выбор региона
2. `borderLayer` — чёрные границы между регионами
3. `selectionLayer` — жёлтый `#fbbf24` контур выбранного региона (только внешние рёбра)
4. `arrowLayer` — миграционные стрелки
5. `labelLayer` — emoji-иконки видов поверх центроида региона

### Стрелки (migrationSmoothed)
- Одна стрелка на пару регионов (top species по суммарному потоку)
- Направление: centroid → ближайший border hex, касающийся региона-получателя
- Стрелка остаётся внутри source-региона, не пересекает границу
- `strokeWidth = clamp(totalCount/100, 0.5, 3.5)`
- `opacity = clamp(totalCount/5, 0, 0.9)`, fade < 0.02 → скрыта
- Наконечник = SVG polygon (не marker, не context-stroke)
- Emoji-лейбл на середине стрелки

### Emoji-лейблы регионов
- Все виды с ≥5% от total biomass региона
- Горизонтальный ряд с spacing=10, центрирован по centroid
- Font-size: 11

### Выбор региона
- Клик → желтый border highlight (только внешние рёбра через hex-соседство)
- `setSelectedRegion` → `updateSelectionHighlight`
- Повторный клик = сброс

---

## 8. Save / Load

`web-sandbox/src/savegame.ts`

**Формат (version 1):**
```json
{
  "version": 1,
  "config": { "worldSeed": "...", ... },
  "tick": 42,
  "regions": [
    { "populations": {"gazelle": 120}, "activeEffects": [], "popHistory": {...} }
  ],
  "globalPopHistory": { "gazelle": [100, 110, ...] }
}
```

**Load:** `generateWorld(config)` → override populations / popHistory / activeEffects per region.

---

## 9. Целевая архитектура (фактическое состояние)

```
web-sandbox/
  src/
    sim/
      types.ts          — WorldState, Region, MigrationEdge, SimConfig
      rng.ts            — rand01, shuffleIndices (SHA256)
      biomes.ts         — 9 биомов
      species.ts        — 8 видов + size/consumption
      worldgen.ts       — Voronoi regions, biome assign, seed species
      simulation.ts     — двухфазовый tick + smoothMigrationEdges
    render/
      hexmap.ts         — SVG hex grid, arrows, labels, selection
      inspector.ts      — боковая панель региона
      chart.ts          — global population chart
      hud.ts            — tick counter, speed buttons, biomass/species HUD
    lab/
      labsim.ts         — изолированный tick с LAB_CONSUMPTION формулой
    food-lab.ts         — food lab UI entry
    main.ts             — основное приложение
    savegame.ts         — save/load JSON
    style.css
  index.html
  food-lab.html
  vite.config.ts        — multi-page build (main + foodLab)
```

---

## 10. Конфиг (SimConfig)

| Параметр | Дефолт | Описание |
|---|---|---|
| `totalHexes` | 500 | Примерное число land-тайлов |
| `numRegions` | 20 | Целевое число регионов |
| `minRegionSize` | 7 | Минимум гексов в регионе |
| `richnessPerHex` | 10 | Ресурсный бюджет на гекс |
| `startPopDensity` | 2 | Стартовая поп на гекс для каждого вида в паке |
| `extinctionEps` | 1.0 | Порог вымирания |
| `worldSeed` | "42" | Сид для всего worldgen |

---

## 11. Открытые вопросы / следующие шаги

1. **Food lab validation** — прогнать все biome packs через food lab с LAB_CONSUMPTION,
   убедиться что все родные комбинации дают стабильное плато без доминирования.
   Подкрутить LAB_CONSUMPTION если нужно.

2. **Интеграция size/consumption в main sim** — когда food lab покажет стабильную формулу,
   перенести в основной `simulation.ts`. До этого — только в lab.

3. **Нормализация видимости стрелок по radiation** — сейчас слон (radiation=2%) редко
   появляется на стрелках даже при доминировании. Возможное решение: нормировать
   `count / radiation` для opacity.

4. **События** — архитектура `activeEffects` есть, `bEnableEvents=false`. Включать после
   стабилизации food mechanics.

5. **Баланс** — проверить что после интеграции consumption динамика выглядит правдоподобно
   (крупные виды давят сильнее, но хуже масштабируются по численности).

### Закрытые задачи (для контекста при регрессии)
- Voronoi вместо BFS-рост: `worldgen.ts` — компактные регионы без "червеобразных" форм
- Arrow flicker → exponential smoothing DECAY=0.92, THRESHOLD=0.1
- Arrow arrowheads → SVG polygon (не marker; маркеры давали размер с strokeWidth)
- Arrow overlap (mutual migration) → каждая стрелка внутри source-региона, не пересекает границу
- Параллельные стрелки (per-species) — опробовано, откатили: слишком шумно
- Save/load: JSON download + FileReader upload
- Multi-species emoji labels ≥5% biomass
- Selection highlight только внешние рёбра региона
