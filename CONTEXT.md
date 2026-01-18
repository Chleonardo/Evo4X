# Evo4X — Project Context (v2.6.x)

## CURRENT BASELINE (for new threads)

As of <18.01.2026>:

- Engine: evo4x_engine_v2_6_13_rtsgrid_uiapi.py
- UI: app.py
- This version is considered STABLE BASELINE for further work.
- Any new changes must be local, minimal, and contract-driven.

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
- Периодически происходят флуктуации ресурсов (ивенты).
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

1. Active effects не висят весь TTL.
2. Consumption share баг.
3. Линейное удорожание трейтов.
4. Форк экономически невыгоден.

---

## 5. Текущий стек

- Python 3.13
- Streamlit UI
- File-based saves

## Active Effects visibility

Goal:
Active effects must be visible in UI for the entire duration (TTL).

Status:
❌ Not fixed yet (baseline version still has early disappearance issue).

Notes:
- Effects exist in engine state with TTL.
- UI sometimes loses them before TTL reaches 0.
- This will be addressed as a single-task patch.

## External playtesting

- Prototype is shared via Streamlit Cloud.
- Testers use browser only (no local setup).
- Onboarding text exists and is sent separately.

