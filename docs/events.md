# Events — Engine Documentation

## Обзор

В Evo4X на каждом тике может произойти случайное событие, которое
накладывает **временный множитель** на один ресурс одной клетки.
После фикса [choose_index](../evo4x_engine_v2_6_13_rtsgrid_uiapi.py#L140)
(апрель 2026) события корректно распределяются между R1/R2/R3 и между клетками.

Сейчас в игре **один тип события**: `fluctuation` (колебание ресурса).
`invasive` удалён.

## Параметры (RULES)

| Ключ                   | Значение | Смысл |
|------------------------|---------:|-------|
| `event_base`           | `0.10`   | Базовый шанс события на тике |
| `event_pity`           | `0.05`   | +шанс за каждый тик без события |
| `event_cap`            | `0.60`   | Потолок шанса |
| `event_fluct_mult_min` | `0.30`   | Минимум множителя (сильная засуха) |
| `event_fluct_mult_max` | `0.70`   | Максимум множителя (лёгкая засуха) |
| `event_fluct_ttl_min`  | `3`      | Минимальный TTL (в тиках) |
| `event_fluct_ttl_max`  | `6`      | Максимальный TTL |

Все множители < 1 → **события всегда вредят** (это "колебание в минус" / засуха).

## Хранилище в save

Активные эффекты живут в `save["state"]["active_effects"]` как список:

```json
{
  "type": "fluctuation",
  "cell": "c01",
  "res":  "R2",
  "mult": 0.47,
  "ttl":  5
}
```

`ticks_since_event` — pity-счётчик.
`last_event_tick` — номер тика последнего события (для UI).
`last_events` — список событий последнего тика (для UI-тоста).

## Последовательность тика (где происходит событие)

Вся логика в [simulate_tick()](../evo4x_engine_v2_6_13_rtsgrid_uiapi.py).
Порядок шагов:

1. **Forks / split** — игрок создаёт новые виды (если action запросил).
2. **Migrations** — `apply_migrations(...)`.
3. **Births** — `n_pre = pop * (1 + r)` для каждого игрового вида.
4. **Event roll** — `event_roll(save_obj)` (см. ниже пошагово).
   - Если сработало `fluctuation`: новый эффект пушится в `state["active_effects"]`
     **до** применения пищи в этом же тике → эффект применяется сразу.
5. **Apply active effects** — `_apply_active_effects(save_obj)` собирает
   словарь `{cell: {res: mult}}` из всего списка `active_effects`.
6. **Allocation + starvation** — для каждой клетки:
   - `eff_res = _effective_resources(cell, active_effects[cell])` — базовые
     ресурсы × множитель.
   - `alloc_food_for_cell(eff_res, contestants)` — пропорциональное деление по `w = n_pre * b[r]`.
   - Выжившие = `min(n_pre, food_allocated)`.
7. **Commit population** — записываем новые популяции, выкидываем вымерших
   (`surv <= extinction_eps`).
8. **Tick down effects** — `_tick_down_effects(save_obj)`: `ttl -= 1`,
   записи с `ttl <= 0` удаляются.
9. **Economy** — EVO за разведку + passive_preview.
10. **Advance tick** — `state.tick += 1`.
11. **Cache for UI** — `last_events`, `last_reveals`, `history`.

> **Важно:** эффект применяется в тот же тик, в котором он возник, и живёт
> ровно `ttl` тиков (т.е. с тика N по тик N+ttl-1 включительно).

## Пошагово: что делает `event_roll`

`event_roll(save_obj) -> (happened: bool, event_info: dict|None)`

1. **Шанс события.**
   ```
   chance = min(event_cap, event_base + event_pity * ticks_since_event)
   roll   = rand01(event_seed, f"tick:{t}:roll")
   if roll >= chance:    # событие НЕ сработало
       ticks_since_event += 1
       return (False, None)
   ```

2. **Сработало → сбрасываем pity-счётчик** и фиксируем `last_event_tick`.

3. **Выбор клетки.** Берутся только клетки, где у игрока есть популяция,
   сортируются, выбираются через `choose_index(..., len(occ))`.
   Если игрок нигде не живёт → `{type: "noop"}`.

4. **Список ресурсов клетки.** `res_list = [R для R в (R1,R2,R3), если base > 0]`.
   - Важно: используется **базовый** `cell["resources"]`, не `effective`, поэтому
     событие на "R2" может возникнуть, даже если прошлое событие высушило R2 до 0.
   - Если вдруг базовые все нулевые → `{type: "noop"}`.

5. **Выбор ресурса.** `res = res_list[choose_index(..., len(res_list))]` —
   **равномерное распределение** по доступным ресурсам клетки.

6. **Параметры колебания.**
   - `mult = lerp(0.30, 0.70, rand01(...))` — насколько сильно режем.
   - `ttl  = int(lerp(3, 6.9999, rand01(...)))` — длительность.

7. **Возвращаем:**
   ```python
   eff = {"type":"fluctuation", "cell":cell, "res":res, "mult":mult, "ttl":ttl}
   return True, {"type":"fluctuation", "eff":eff, "cell":..., "res":..., "mult":..., "ttl":...}
   ```

## Применение эффекта (две ключевые функции)

### `_apply_active_effects(save_obj) -> {cell: {res: mult}}`
Перемножает множители одной и той же пары (cell, res), если активно несколько
эффектов на один и тот же ресурс.

### `_effective_resources(cell, effects_cell) -> {R1, R2, R3}`
```python
res = dict(cell["resources"])
for r, mult in effects_cell.items():
    res[r] = res[r] * mult
return res
```
Результат уходит в `alloc_food_for_cell` как пул еды для этого тика.

## Отображение в UI

- **HUD caption** (`render_hud`, [app.py:139](../app.py#L139)) —
  `⚡ c01 🌿×0.47(5t)`: цельный список активных эффектов.
- **Карта** (`render_map`, [app.py:148](../app.py#L148)) — на клетке с эффектом
  добавляется `⚡` к названию архетипа и ресурсы показываются как
  `resources_effective` (с учётом множителя).
- **Cell inspector** (`render_cell`) — баннер `⚡ Event — 🌿 Leaves ×0.47 (5 ticks left)`
  плюс базовые/эффективные значения ресурсов с разницей.

## Исторический баг (fixed)

До апреля 2026 в `choose_index` была формула
`int(lerp(0, n-1+1e-9, rand01))`. Для `n=2` она всегда возвращала `0`, для `n=3`
— `0` или `1`, но никогда `2`. Из-за этого:
- события всегда падали на R1;
- клетки выбирались смещённо в сторону начала отсортированного списка;
- worldgen-шаблоны/повороты/pack'ы тоже были смещены.

Фикс: `min(n-1, int(rand01 * n))` — равномерное распределение по `[0, n)`.

## Примечания для будущих правок

- Эффекты кумулятивны: на один ресурс одной клетки можно повесить несколько
  колебаний — множители перемножатся.
- Базовые ресурсы (`cell["resources"]`) **никогда не меняются** во время тика —
  весь эффект живёт в `active_effects`. Это позволяет легко откатываться и
  показывать в UI "base → effective".
- Если захочется "положительных" колебаний — просто расширьте
  `event_fluct_mult_max` выше 1.0. Никакой логики применения менять не надо.
