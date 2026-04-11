# Evo4X — Project Context (v2.6.x)

## CURRENT BASELINE (for new threads)

As of <11.04.2026>:

- Engine: evo4x_engine_v2_6_13_rtsgrid_uiapi.py
- UI: app.py
- This version is considered STABLE BASELINE for further work.
- Any new changes must be local, minimal, and contract-driven.
- See [docs/events.md](docs/events.md) for the full event-system spec.

## 0. Что это за документ
Этот файл — единый источник правды для контекста проекта Evo4X.
Он предназначен для:
- переноса контекста между чатами / сессиями,
- быстрого онбординга в проект,
- фиксации текущих правил, гипотез и известных проблем.

## Change policy

- No refactors without explicit request.
- No renaming of public fields or functions.
- All changes must be single-task and contract-based.
- Prefer patching over rewriting.
- Engine is the source of truth; UI only reads snapshot.


---

## 1. Описание игры (человеческое + геймдизайнерское)

### Коротко
Evo4X — это пошаговая эволюционная 4X-стратегия про рост вида в экосистеме,
где игрок управляет не юнитами, а эволюционными решениями вида.

Игрок:
- распределяет популяцию по клеткам мира,
- конкурирует с NPC-видами за ресурсы,
- покупает эволюционные трейты,
- решает, когда выгоднее улучшать вид, а когда — форкать его.

### Роль игрока
Игрок — эволюционный архитектор, а не микроменеджер.

Он не управляет:
- конкретными особями,
- производством,
- тактическими боями.

Он управляет:
- скоростью размножения,
- специализацией потребления ресурсов,
- пространственным распространением вида,
- стратегией выживания в условиях ограниченной среды.

### Фантазия / нарратив
- Мир — абстрактная экосистема, разбитая на клетки.
- В каждой клетке есть ресурсы (R1, R2, R3).
- Есть NPC-виды, которые эволюционируют и конкурируют за ресурсы.
- Периодически происходят флуктуации ресурсов (ивенты) — единственный тип
  события на данный момент; invasive удалён в апреле 2026.
- Игрок не знает будущего заранее, но может адаптироваться.

---

## 2. Структура мира

### Карта
- Размер: 3×3 клетки (c00 … c22)
- Каждая клетка имеет:
  - соседей,
  - ресурсы,
  - NPC-популяцию,
  - популяцию игрока.

---

## 3. Экономика и эволюция

- Evo — абстрактная валюта эволюции.
- Начисляется каждый тик.
- Тратится на трейты и форк.

---

## 4. Известные проблемы

1. ~~Active effects не висят весь TTL.~~ → fixed (2026-04-11, см. ниже).
2. Consumption share баг.
3. Линейное удорожание трейтов.
4. Форк экономически невыгоден.

### Fixed — 2026-04-11

- **`choose_index` всегда возвращал 0 для `n=2`** (и максимум `n-2` для больших `n`).
  Формула `int(lerp(0, n-1+1e-9, rand01))` математически неверна. Следствия:
  события падали только на R1, worldgen-шаблоны/повороты/pack'ы были смещены,
  Fisher-Yates давал вырожденный shuffle. Пруф — в 600-тиковом сейве было
  141 событие, все на R1, при том что 66 из них катились на клетках с
  доступными R2/R3. Фикс: `min(n-1, int(rand01 * n))`.
- **Invasive events removed.** Остался только `fluctuation`. Код и ключи
  `event_invasive_*`, `npc_b_focus_*`, `npc_b_other_*` удалены из RULES/event_roll.

---

## 5. Текущий стек

- Python 3.13
- Streamlit UI
- File-based saves

## Active Effects visibility

Goal:
Active effects must be visible in UI for the entire duration (TTL).

Status:
✅ Fixed (2026-04-11). Первопричиной был не UI, а `choose_index` —
события почти всегда приходились на R1, из-за чего R2/R3-эффекты
выглядели "пропавшими". После фикса RNG они корректно висят весь TTL
и отображаются в HUD / на карте / в cell inspector.

Подробности и контракт системы событий — в [docs/events.md](docs/events.md).

## External playtesting

- Prototype is shared via Streamlit Cloud.
- Testers use browser only (no local setup).
- Onboarding text exists and is sent separately.

