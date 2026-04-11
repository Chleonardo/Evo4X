# Evo4X — Project Context

> **Назначение этого файла.** Единый источник правды о проекте для быстрого онбординга
> новой ИИ-сессии или человека. Что это за игра, как устроен код, какие инварианты,
> где копать дальше. Только актуальное состояние — без истории исправленных багов.

**Baseline (последнее обновление: 2026-04-11)**
- Engine: [evo4x_engine_v2_6_13_rtsgrid_uiapi.py](evo4x_engine_v2_6_13_rtsgrid_uiapi.py)
- UI: [app.py](app.py) (Streamlit)
- Data: [data/species.json](data/species.json), [data/archetypes.json](data/archetypes.json)
- Tests: [validate_worldgen.py](validate_worldgen.py)
- Event-system spec: [docs/events.md](docs/events.md)
- Prod: https://evo4xgame.streamlit.app/ (авто-деплой из `master`)

---

## 1. Что это за игра

**Evo4X** — пошаговая эволюционная 4X-стратегия. Игрок управляет не юнитами, а
**эволюционными решениями одного (или нескольких) видов** в экосистеме 3×3 клеток.

### Что игрок делает
- Распределяет популяцию между клетками (миграции).
- Покупает **трейты** (+r, +b1/b2/b3) за ресурс `EVO`.
- Может **форкнуть** вид — отщепить часть популяции в новый, самостоятельно эволюционирующий вид.
- Конкурирует с **NPC-видами** (сеяные worldgen'ом) за ресурсы.
- Адаптируется к случайным **событиям** (флуктуациям ресурсов).

### Что игрок НЕ делает
- Не управляет отдельными особями, боями, постройками.
- Не видит будущего — все решения на основе текущего состояния.

### Фантазия
Абстрактная саванна. Каждая клетка — биом со своим профилем ресурсов. Вид, который
эффективнее использует доступный ресурсный микс, побеждает. Но профиль можно
изменить трейтами и пространственной экспансией.

---

## 2. Мир

### Карта
- **3×3** клетки с id `cXY` (X — колонка, Y — ряд, обе 0-indexed).
- Старт всегда в `c11` (центр).
- Клетки соединены 4-соседством (без диагоналей).

### Ресурсы (3 типа)
| ID | UI name | Emoji |
|----|---------|-------|
| R1 | Grass   | 🌱    |
| R2 | Leaves  | 🌿    |
| R3 | Roots   | 🥔    |

### Архетипы клеток
9 архетипов задают соотношение R1/R2/R3 — см. [data/archetypes.json](data/archetypes.json):
`grassland`, `open_savanna`, `woodland`, `dense_grove`, `root_patch`, `rootland`,
`diverse_savanna`, `root_mosaic`, `leaf_mosaic`. Например, grassland = 100% R1,
dense_grove = 20/80/0, root_mosaic = 40/20/40.

### Richness (богатство клетки)
`poor=30`, `normal=60`, `rich=90` — суммарный пул ресурсов клетки. Реальные R1/R2/R3
получаются через `archetype.ratio × richness_total`.

### Worldgen
[_savanna_worldgen()](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L482) — полностью
детерминированный по `world_seed`:
- Start (c11): всегда `grassland/poor`, без NPC.
- 4 соседа c01/c10/c12/c21: раскладка слотов (`safe_training`, `leaf_offspec`,
  `root_offspec`, `wildcard`) через перестановку + случайный выбор варианта внутри
  слота. Соседи никогда не `rich`.
- 4 угла c00/c02/c20/c22: шаблон + поворот + отражение + подбор richness+pack.
- Инварианты (проверяются [validate_worldgen.py](validate_worldgen.py)):
  - max 2 rich клетки на мир, 2–3 poor клетки (включая start).
  - Rich углы обязаны иметь "contested" пак (buffalo, elephant или ≥3 видов).
  - Все комбинации archetype×pack стабильны 500 тиков в NPC-only симуляции.

### Виды (species)
8 сеяных видов — см. [data/species.json](data/species.json):
`gazelle`, `impala`, `zebra`, `buffalo`, `giraffe`, `elephant`, `warthog`, `mole_rat`.
У каждого `r` (прирост) и `b = {R1,R2,R3}` (вес потребления по ресурсам).

NPC-виды сеются из паков по archetype/richness с фиксированными стартовыми
популяциями из `SAVANNA_NPC_POP`.

---

## 3. Мат. модель тика

### Порядок шагов в [simulate_tick()](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L798)
1. **Passive EVO credit** — `evo += k * sqrt(total_biomass)` в начале тика.
2. **Buy trait** (опциональный `action.buy_trait`) — тратит EVO, `+0.1` к соответствующему стату.
3. **Fork** (опциональный `action.fork`) — тратит EVO, отщепляет популяцию в новый `spN`.
4. **Migrations** (`action.migrations`) — до 50% популяции клетки-источника.
5. **Births** — для каждого игрового вида: `n_pre = pop * (1 + r)` в каждой клетке.
6. **Event roll** — см. [docs/events.md](docs/events.md). Новый эффект кладётся в
   `state.active_effects` и применяется в этот же тик.
7. **Apply active effects** — собирает `{cell: {res: mult}}`.
8. **Allocation + starvation** — для каждой клетки:
   - `eff_res = base × multiplier_от_эффектов`
   - `alloc_food_for_cell(eff_res, contestants)` — пропорциональное деление
     по `w = n_pre × b[r]` для каждого ресурса по очереди.
   - `survivors = min(n_pre, food_allocated)`.
9. **Commit populations** — новые значения; вымершие (`surv ≤ extinction_eps=1.0`) удаляются.
10. **Tick down effects** — `ttl -= 1`, истёкшие удаляются.
11. **Economy** — EVO за разведку новых клеток (+3 за клетку, разово).
12. **Advance tick number**.
13. **History/cache** — `last_events`, `last_reveals`, `pop_history`, `history`.

### Формулы
- Рост вида: `n_pre(cell) = pop(cell) × (1 + r)`.
- Голод: `survivors = min(n_pre, food_allocated)`. Лишнее съеденное — не сохраняется.
- Распределение еды по ресурсу R: `food[i] += avail[R] × w[i] / Σw`, где `w[i] = n_pre[i] × b[i][R]`.
- Passive EVO: `passive = evo_passive_k × sqrt(total_player_biomass)` = `0.4 × √B`.
- Trait cost: **нелинейный**, `base 3.0 + (2 + 3 + 4.5 + 6.75 + …)` с шагом ×1.5. См. [trait_cost()](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L192).
- Fork cost: **нелинейный**, `15 × 1.7^(n_species - 1)`. См. [fork_cost()](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L207).
- Extinction threshold: `extinction_eps = 1.0` (единый для игрока и NPC).
- Migration cap: до 50% популяции клетки-источника.
- Min expedition: 2.0 (минимум особей при форке/миграции).

### События
Единственный тип: `fluctuation`. Множитель `0.30–0.70` (т.е. всегда "засуха"),
TTL `3–6` тиков. Полный контракт в [docs/events.md](docs/events.md).

---

## 4. Архитектура кода

### Разделение
- **Engine** — вся игровая логика, ничего про UI. Чтение/запись сейвов (JSON-файлы в `runs/`).
- **UI (app.py)** — Streamlit; работает только через `get_ui_snapshot()` и `simulate_tick()`.
- **Data (data/*.json)** — справочники видов и архетипов. Загружаются при import.

**Инвариант:** UI никогда не лезет во внутренности save'а, только через API движка.
Это контракт: можно менять любое внутреннее поле save-файла, если снапшот остался тем же.

### Основной API движка
| Функция | Что делает |
|---|---|
| `init_new_run(world_seed, npc_seed, event_seed, card_seed, out_dir)` | Создаёт новый run, возвращает `(save_path, hud)` |
| `simulate_tick(save_path, action, out_dir)` | Применяет action и один тик, возвращает `(new_save_path, result)` |
| `get_ui_snapshot(save_path, selected_cell=None)` | Единый snapshot для рендера UI |
| `get_hud` / `get_economy_cached` / `get_species_list` / `get_species_cells` | Точечные API (вызываются внутри `get_ui_snapshot`) |
| `get_cell_inspector` / `get_cell_passport` / `get_cell_consumption` | По-клеточные детали |

### Формат action
```python
{
    "buy_trait": ("sp0", "r" | "b1" | "b2" | "b3"),   # опционально
    "fork":      ("sp0", amount_or_"ALL", "r"|"b1"|"b2"|"b3", source_cell),  # опц.
    "migrations": [("sp0", "c11", "c01", 2.0), ...],  # опц.
}
```

### Формат save (сокращённо)
```
meta:    { save_id, engine }
rng:     { model: "sha256_stateless", world_seed, npc_seed, event_seed, card_seed }
world:   { w, h, cells: {cid: {archetype, richness, resources, npc_species, neighbors}}, start_cell }
player:  { evo, species: {sid: {stats:{r,b}, traits:{r,b1,b2,b3}, population:{cid:pop}}}, explored }
state:   {
    tick, active_effects, ticks_since_event, scouted_cells,
    last_tick_breakdown, history, pop_history, pop_history_total,
    last_events, last_reveals, hud, economy, passive_preview, selected_cell
}
```

### Структура app.py
- `ensure_state / safe_snapshot` — state и защищённый доступ к snapshot'у.
- `do_new_game / do_load_game / do_tick` — мутации через движок.
- `render_hud` — верхняя панель + active effects caption.
- `render_map` — 3×3 сетка клеток с кнопками, per-species иконки и трендовые стрелки.
- `render_cell` — панель выбранной клетки (события, ресурсы, NPC, игроки, consumption).

### RNG
Стейтлесс sha256: `rand01(seed, key) = sha256("{seed}|{key}")[:8] as uint32 / 2^32`.
Любой детерминированный выбор должен использовать `rand01 / choose_index`, **не**
`random.*`. `choose_index(seed, key, n) = min(n-1, int(rand01 × n))`.

### Данные (не трогать без нужды)
[data/species.json](data/species.json) и [data/archetypes.json](data/archetypes.json)
— единственный способ тюнить виды/биомы. Движок читает их при import'е через
`_load_json`. Они закоммичены в репу (см. `.gitignore` `!data/*.json`).

---

## 5. Change policy

Как писать код в этом репозитории:

- **No refactors without explicit request.** Минимальные точечные патчи.
- **No renaming public fields or functions** — UI зависит от контракта snapshot'а.
- **Engine is source of truth.** UI только читает snapshot и шлёт `action`-dict.
- **Preserve determinism.** Любая новая случайность — только через `rand01 / choose_index`
  с осмысленным ключом. Один и тот же `world_seed` обязан давать идентичный мир.
- **Prefer patching over rewriting.** Read first → edit targeted region → verify.
- **Никакой документации/комментариев без запроса.**
  Только там, где явная непонятная логика.
- **Тесты worldgen'а должны остаться зелёными:**
  `PYTHONIOENCODING=utf-8 python validate_worldgen.py` → `ALL TESTS PASSED`.

---

## 6. Текущие открытые вопросы / известные проблемы

*(живой список — обновлять при обнаружении/закрытии)*

1. **Consumption share visualization** — доли потребления в UI местами не
   совпадают с пропорциями allocation'а. Надо сверить, что UI берёт `food_eaten /
   cell_capacity_total` из `last_tick_breakdown`, а не пересчитывает сам.
2. **Баланс NPC-паков vs игрок.** Smoke-тест (`validate_worldgen.py [6]`)
   показывает, что в одиночных столкновениях с `giraffe` / `gazelle` / mixed-паками
   стартовый `sp0 {r=0.5, b=[1, 0.5, 0]}` часто коллапсит к 200 тикам. Это может
   быть WAI (нужна экспансия/трейты), но хочется это валидировать после выката в плейтест.
3. **UX онбординга.** Плейтестеры в Streamlit Cloud — что именно им непонятно
   при первом запуске, пока не систематизировано.

### Закрытые в этой ветке (для контекста, чтобы знать где смотреть при регрессии)
- `choose_index` RNG bias → фикс в [engine:132](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L132).
- Invasive events удалены целиком.
- Active effects TTL / R2-R3 visibility → следствие фикса `choose_index`.
- Corner richness budget → `_pick_corner_richness` теперь учитывает poor-соседей.
- Trait / fork pricing уже нелинейные (`×1.5` и `×1.7`), не линейные.
- Per-species trend arrows, migration half-bug, effective resources on tiles,
  extinct NPC cleanup — починены в app.py / движке в прошлых сессиях.

---

## 7. Стек и окружение

- **Python 3.13**
- **Streamlit** (UI)
- File-based JSON сейвы в `runs/` (локально и на Streamlit Cloud временно;
  перезагрузка инстанса их теряет — это known, не фичепожелание)
- Git: [github.com/Chleonardo/Evo4X](https://github.com/Chleonardo/Evo4X), ветка `master`.
- Прод: https://evo4xgame.streamlit.app/ — автодеплой из `master` через ~30–60 сек
  после push. При смене `requirements.txt` нужен ручной **Reboot app** в
  Streamlit Cloud админке.

---

## 8. Куда смотреть дальше

- **Как работает один тик** → [evo4x_engine_v2_6_13_rtsgrid_uiapi.py:798 simulate_tick](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L798)
- **Как работают события** → [docs/events.md](docs/events.md)
- **Как устроен worldgen** → [_savanna_worldgen()](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L482) + [validate_worldgen.py](validate_worldgen.py)
- **Что видит UI** → [get_ui_snapshot()](evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L1287) и [app.py render_map/render_cell](app.py#L148)
- **Балансные константы** → `RULES` dict в топе движка + [data/*.json](data/)
