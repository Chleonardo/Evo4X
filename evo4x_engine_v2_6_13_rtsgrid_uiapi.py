# Evo4X Engine v2.6 (Streamlit-friendly)
# - Deterministic RNG via sha256 (stateless)
# - simulate_tick returns dicts: HUD + reveals + events + economy
# - Reveal cell passport AFTER the tick resolves (state at that moment)
# - EVO passive credited at START of tick from stored passive_due
# - Migrations are species-aware

import os, json, math, time, hashlib

RULES = {
    'extinction_eps': 0.01,
    "map_w": 3,
    "map_h": 3,
    "min_expedition": 2.0,
    "migration_cap_frac": 0.5,

    "explore_reward": 3.0,          # one-time, first time you ever enter a cell (migration destination)
    "evo_passive_k": 0.4,           # passive_due_next = k*sqrt(total_biomass_post_starvation)

    "event_base": 0.10,
    "event_pity": 0.05,
    "event_cap": 0.60,

    "event_fluct_mult_min": 0.30,
    "event_fluct_mult_max": 0.70,
    "event_fluct_ttl_min": 3,
    "event_fluct_ttl_max": 6,

    "event_invasive_pop_min": 10.0,
    "event_invasive_pop_max": 25.0,

    "npc_b3_cap": 0.30,
    "npc_r_min": 0.10,
    "npc_r_max": 0.60,

    "npc_b_focus_min": 1.20,
    "npc_b_focus_max": 2.00,
    "npc_b_other_min": 0.00,
    "npc_b_other_max": 0.60,
}

# ---------- RNG (sha256_stateless) ----------

def _u32(seed_int: int, key: str) -> int:
    s = f"{int(seed_int)}|{key}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def rand01(seed_int: int, key: str) -> float:
    return _u32(seed_int, key) / 2**32

def lerp(a, b, t):
    return a + (b - a) * t

def choose_index(seed_int: int, key: str, n: int) -> int:
    if n <= 0:
        return 0
    return int(lerp(0, n - 1 + 1e-9, rand01(seed_int, key)))

# ---------- Save IO ----------

def read_save(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_save(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- Economy ----------

def _traits_counters(spec: dict) -> dict:
    """Normalize trait storage to a counter dict.

    Accepts:
      - spec["traits"] as dict counters, e.g. {"r":2,"b1":1}
      - legacy per-trait flat keys on spec (r,b1,b2,b3)
      - legacy list of strings like ["b2+0.1", ...]
    """
    out = {"r": 0, "b1": 0, "b2": 0, "b3": 0}

    raw = spec.get("traits")
    if isinstance(raw, dict):
        for k in out:
            try:
                out[k] = int(raw.get(k, 0))
            except Exception:
                out[k] = 0
        return out

    # legacy: stored as flat keys on spec
    legacy_keys = any(k in spec for k in ("r", "b1", "b2", "b3"))
    if legacy_keys:
        for k in out:
            try:
                out[k] = int(spec.get(k, 0) or 0)
            except Exception:
                out[k] = 0
        return out

    # legacy: list of strings
    if isinstance(raw, list):
        for t in raw:
            if not isinstance(t, str):
                continue
            for k in out:
                if t.startswith(k):
                    out[k] += 1
        return out

    return out


def trait_cost(spec: dict) -> float:
    """Cost of the NEXT trait purchase for a species spec.

    Uses nonlinear increments: +2, +3, +4.5... (x1.5 each step), on top of base 3.0.
    """
    base = 3.0
    traits_total = sum(_traits_counters(spec).values())
    inc0 = 2.0
    growth = 1.5
    add = 0.0
    for i in range(traits_total):
        add += inc0 * (growth ** i)
    return round(base + add, 1)


def fork_cost(n_species: int) -> float:
    return 15.0 * (1.7 ** max(0, n_species - 1))

# ---------- Core helpers ----------

def births(pop_map: dict, r: float) -> dict:
    return {c: float(n) * (1.0 + float(r)) for c, n in pop_map.items()}

def total_player_biomass(save_obj: dict) -> float:
    tot = 0.0
    for sp in save_obj["player"]["species"].values():
        for n in sp["population"].values():
            tot += float(n)
    return tot

def _apply_active_effects(save_obj: dict) -> dict:
    effects = {}
    for eff in save_obj["state"].get("active_effects", []):
        cell = eff["cell"]
        res = eff["res"]
        mult = float(eff["mult"])
        effects.setdefault(cell, {})
        effects[cell][res] = effects[cell].get(res, 1.0) * mult
    return effects

def _tick_down_effects(save_obj: dict) -> None:
    keep = []
    for eff in save_obj["state"].get("active_effects", []):
        eff["ttl"] = int(eff["ttl"]) - 1
        if eff["ttl"] > 0:
            keep.append(eff)
    save_obj["state"]["active_effects"] = keep

def _effective_resources(cell: dict, effects_cell: dict) -> dict:
    res = {k: float(v) for k, v in cell["resources"].items()}
    for r, mult in effects_cell.items():
        res[r] = res.get(r, 0.0) * float(mult)
    return res

def alloc_food_for_cell(eff_resources: dict, contestants: list) -> dict:
    # contestants: list of dicts {id, kind, n_pre, b:{R1,R2,R3}}
    food = {c["id"]: 0.0 for c in contestants}
    for r in ("R1", "R2", "R3"):
        avail = float(eff_resources.get(r, 0.0))
        if avail <= 0:
            continue

        sumw = 0.0
        weights = []
        for c in contestants:
            w = float(c["n_pre"]) * float(c["b"].get(r, 0.0))
            weights.append((c["id"], w))
            sumw += w

        if sumw <= 1e-12:
            continue

        for cid, w in weights:
            food[cid] += avail * (w / sumw)

    return food

# ---------- NPC focus (for UI) ----------

def _npc_focus(b: dict) -> str:
    best = "R1"
    bestv = float(b.get("R1", 0.0))
    for r in ("R2", "R3"):
        v = float(b.get(r, 0.0))
        if v > bestv:
            best, bestv = r, v
    return best

# ---------- Passport (cell inspect) ----------

def _make_cell_passport(save_obj: dict, cell_name: str, tick_resolved: int) -> dict:
    cell = save_obj["world"]["cells"][cell_name]
    effects = _apply_active_effects(save_obj).get(cell_name, {})
    eff_res = _effective_resources(cell, effects)

    npcs = []
    for i, npc in enumerate(cell.get("npc_species", [])):
        npcs.append({
            "npc_id": f"npc{i}",
            "pop": float(npc["pop"]),
            "r": float(npc["r"]),
            "b": {k: float(v) for k, v in npc["b"].items()},
            "focus": _npc_focus(npc["b"]),
            "origin": npc.get("origin", "seed"),
        })

    players = []
    for sid, sp in save_obj["player"]["species"].items():
        if cell_name in sp["population"]:
            players.append({
                "species_id": sid,
                "pop": float(sp["population"][cell_name]),
                "r": float(sp["stats"]["r"]),
                "b": {k: float(v) for k, v in sp["stats"]["b"].items()},
            })

    breakdown = save_obj["state"].get("last_tick_breakdown", {}).get(cell_name)

        # --- Active effects (detailed, with TTL) ---
    active_effects_detailed = []
    for eff in (save_obj.get("state", {}).get("active_effects", []) or []):
        try:
            if eff.get("cell") != cell_name:
                continue
            active_effects_detailed.append({
                "type": str(eff.get("type", "fluctuation")),
                "cell": str(eff.get("cell")),
                "res": str(eff.get("res")),
                "mult": float(eff.get("mult")),
                "ttl": int(eff.get("ttl")),
            })
        except Exception:
            # ignore malformed effect entries
            continue

    return {
        "cell": cell_name,
        "tick": int(tick_resolved),
        "resources": {k: float(v) for k, v in cell["resources"].items()},
        "resources_effective": {k: float(v) for k, v in eff_res.items()},
        "active_effects": [{"res": k, "mult": float(v)} for k, v in effects.items()],
        "active_effects_detailed": active_effects_detailed,
        "npc_species": npcs,
        "player_species": players,
        "consumption_breakdown": breakdown,
    }

# ---------- Events ----------

def event_roll(save_obj):
    # Event pity-timer stored in state["ticks_since_event"] (int)
    tse = int(save_obj["state"].get("ticks_since_event", 0))
    chance = min(RULES["event_cap"], RULES["event_base"] + RULES["event_pity"] * tse)

    t = int(save_obj["state"]["tick"]) + 1
    roll = rand01(save_obj["rng"]["event_seed"], f"tick:{t}:roll")
    if roll >= chance:
        save_obj["state"]["ticks_since_event"] = tse + 1
        return False, None

    save_obj["state"]["ticks_since_event"] = 0
    save_obj["state"]["last_event_tick"] = t
    # 0 = fluctuation, 1 = invasive
    et = choose_index(save_obj["rng"]["event_seed"], f"tick:{t}:etype", 2)

    # choose occupied cell (any player presence)
    occ = set()
    for sp in save_obj["player"]["species"].values():
        for c in sp["population"].keys():
            occ.add(c)
    occ = sorted(list(occ))
    if not occ:
        return True, {"type": "noop"}

    cell = occ[choose_index(save_obj["rng"]["event_seed"], f"tick:{t}:cell", len(occ))]
    res_list = ["R1", "R2", "R3"]
    res = res_list[choose_index(save_obj["rng"]["event_seed"], f"tick:{t}:res:{cell}", 3)]

    if et == 0:
        mult = lerp(RULES["event_fluct_mult_min"], RULES["event_fluct_mult_max"],
                    rand01(save_obj["rng"]["event_seed"], f"tick:{t}:mult:{cell}:{res}"))
        ttl = int(lerp(RULES["event_fluct_ttl_min"], RULES["event_fluct_ttl_max"] + 0.9999,
                       rand01(save_obj["rng"]["event_seed"], f"tick:{t}:ttl:{cell}:{res}")))
        eff = {"type": "fluctuation", "cell": cell, "res": res, "mult": float(mult), "ttl": ttl}
        return True, {"type": "fluctuation", "eff": eff, "cell": cell, "res": res, "mult": float(mult), "ttl": ttl}

    # invasive
    pop = lerp(RULES["event_invasive_pop_min"], RULES["event_invasive_pop_max"],
               rand01(save_obj["rng"]["event_seed"], f"tick:{t}:invpop:{cell}:{res}"))

    b = {"R1": 0.0, "R2": 0.0, "R3": 0.0}
    for r in ("R1", "R2", "R3"):
        if r == res:
            if r == "R3":
                b[r] = float(RULES["npc_b3_cap"])
            else:
                b[r] = float(lerp(RULES["npc_b_focus_min"], RULES["npc_b_focus_max"],
                                  rand01(save_obj["rng"]["event_seed"], f"tick:{t}:bf:{cell}:{res}")))
        else:
            b[r] = float(lerp(RULES["npc_b_other_min"], RULES["npc_b_other_max"],
                              rand01(save_obj["rng"]["event_seed"], f"tick:{t}:bo:{cell}:{r}")))

    b["R3"] = min(float(b["R3"]), float(RULES["npc_b3_cap"]))
    rr = float(lerp(RULES["npc_r_min"], RULES["npc_r_max"],
                    rand01(save_obj["rng"]["event_seed"], f"tick:{t}:invr:{cell}")))
    npc = {"pop": float(pop), "r": float(rr), "b": b, "origin": "invasive"}
    return True, {"type": "invasive", "cell": cell, "res": res, "npc": npc}

# ---------- World generation ----------

def _gen_resources(world_seed: int, cell_name: str, total_min: float, total_max: float, r3_weight: float = 1/3):
    total = lerp(total_min, total_max, rand01(world_seed, f"total:{cell_name}"))
    u1 = rand01(world_seed, f"u1:{cell_name}") + 1e-6
    u2 = rand01(world_seed, f"u2:{cell_name}") + 1e-6
    u3 = (rand01(world_seed, f"u3:{cell_name}") + 1e-6) * r3_weight
    s = u1 + u2 + u3
    return {"R1": float(total * (u1 / s)), "R2": float(total * (u2 / s)), "R3": float(total * (u3 / s))}



def _cell_difficulty(world_seed: int, cell_name: str) -> float:
    # 0..1, used to scale NPC pressure per cell
    return float(rand01(world_seed, f"celldiff:{cell_name}"))

def _npc_count(npc_seed: int, cell_name: str, difficulty: float) -> int:
    # target mean ~1.7, scaled by per-cell difficulty
    r = rand01(npc_seed, f"npccount:{cell_name}")
    r2 = min(0.999999, max(0.0, r + 0.35*(difficulty-0.5)))
    if r2 < 0.45:
        return 1
    if r2 < 0.90:
        return 2
    return 3

def _gen_npc_species(npc_seed: int, cell_name: str, idx: int, difficulty: float) -> dict:
    rr = float(lerp(RULES["npc_r_min"], RULES["npc_r_max"], rand01(npc_seed, f"npcr:{cell_name}:{idx}")))
    b1 = float(lerp(0.2, 2.0, rand01(npc_seed, f"npcb1:{cell_name}:{idx}")))
    b2 = float(lerp(0.2, 2.0, rand01(npc_seed, f"npcb2:{cell_name}:{idx}")))
    b3 = float(lerp(0.0, RULES["npc_b3_cap"], rand01(npc_seed, f"npcb3:{cell_name}:{idx}")))
    pop_base = float(lerp(6.0, 28.0, rand01(npc_seed, f"npcpop:{cell_name}:{idx}")))
    mult = float(0.35 + 0.90 * difficulty)
    pop = float(pop_base * mult)
    return {"pop": pop, "r": rr, "b": {"R1": b1, "R2": b2, "R3": b3}, "origin": "seed"}

def init_new_run(world_seed=0, npc_seed=1, event_seed=2, card_seed=3, out_dir="/content"):
    # Build a 3x3 grid world. Cell ids: cXY (X col, Y row), 0-indexed.
    w = int(RULES.get("map_w", 3))
    h = int(RULES.get("map_h", 3))

    all_cells = [f"c{x}{y}" for y in range(h) for x in range(w)]

    # Random start position
    start_idx = choose_index(world_seed, "startpos", len(all_cells))
    start_cell = all_cells[int(start_idx)]

    cells = {}
    # Create cells with resources + difficulty
    for cid in all_cells:
        if cid == start_cell:
            resources = {"R1": 10.0, "R2": 0.0, "R3": 0.0}
        else:
            resources = _gen_resources(world_seed, cid, 0, 100, r3_weight=1/3)

        diff = _cell_difficulty(world_seed, cid)
        cells[cid] = {
            "pos": {"x": int(cid[1]), "y": int(cid[2])},
            "difficulty": float(diff),
            "resources": resources,
            "npc_species": [],
            "neighbors": [],
        }

    # Neighbors (4-neighborhood) for later RTS rules/UI
    for cid in all_cells:
        x = int(cid[1]); y = int(cid[2])
        nbrs = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                nbrs.append(f"c{nx}{ny}")
        cells[cid]["neighbors"] = nbrs

    # Spawn NPCs (skip start cell)
    for cid in all_cells:
        if cid == start_cell:
            continue
        diff = float(cells[cid]["difficulty"])
        n = _npc_count(npc_seed, cid, diff)
        for i in range(n):
            cells[cid]["npc_species"].append(_gen_npc_species(npc_seed, cid, i, diff))

    save_id = f"seed_{int(time.time())}"
    save_obj = {
        "meta": {"save_id": save_id, "engine": "Evo4X v2.6.7"},
        "rng": {"model": "sha256_stateless", "world_seed": int(world_seed), "npc_seed": int(npc_seed),
                "event_seed": int(event_seed), "card_seed": int(card_seed)},
        "world": {"w": w, "h": h, "cells": cells, "start_cell": start_cell},
        "player": {
            "evo": 0.0,
            "species": {
                "sp0": {
                    "stats": {"r": 0.5, "b": {"R1": 1.0, "R2": 0.5, "R3": 0.0}},
                    "traits": {"r": 0, "b1": 0, "b2": 0, "b3": 0},
                    "population": {start_cell: 2.0},
                }
            },
            "explored": [start_cell],
        },
        "state": {
            "tick": 0,
            "active_effects": [],
            "ticks_since_event": 0,
            # Cells the player has ever entered (for one-time exploration rewards & UI).
            "scouted_cells": [start_cell],
            "last_tick_breakdown": {},
            "history": [],
            "pop_history": {},  # species_id -> cell_id -> [pop_by_tick]
            "selected_cell": start_cell,
        },
    }

    # initialize passive preview for UI (between-ticks)
    save_obj["state"]["passive_preview"] = float(RULES["evo_passive_k"] * math.sqrt(max(0.0, total_player_biomass(save_obj))))
    # initialize cached HUD/economy for UI (avoid UI reading internals)


    # total (player) pop history per cell for map trend
    if "pop_history_total" not in save_obj["state"]:
        save_obj["state"]["pop_history_total"] = {}
    # aggregate player pop by cell
    tot_by_cell = {}
    for _sid, _sp in save_obj["player"]["species"].items():
        for _c, _n in _sp.get("population", {}).items():
            tot_by_cell[_c] = tot_by_cell.get(_c, 0.0) + float(_n)
    for _c, _n in tot_by_cell.items():
        save_obj["state"]["pop_history_total"].setdefault(_c, []).append(float(_n))
        if len(save_obj["state"]["pop_history_total"][_c]) > 30:
            save_obj["state"]["pop_history_total"][_c] = save_obj["state"]["pop_history_total"][_c][-30:]

    save_obj["state"]["hud"] = build_hud(save_obj)
    save_obj["state"]["economy"] = build_economy(save_obj)
    # initialize population history for trend arrows
    for sid, sp in save_obj["player"]["species"].items():
        for cell_id, pop in sp.get("population", {}).items():
            save_obj["state"]["pop_history"].setdefault(sid, {}).setdefault(cell_id, []).append(float(pop))
    save_obj["state"]["selected_cell"] = start_cell


    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_{save_id}.json")
    write_save(path, save_obj)

    return path, build_hud(save_obj)
def apply_migrations(save_obj, migrations):
    if not migrations:
        return set()

    species = save_obj["player"]["species"]
    touched_dst = set()

    for (sid, src, dst, amt) in migrations:
        if sid not in species:
            raise ValueError("Unknown species in migration: " + str(sid))
        amt = float(amt)
        if amt < RULES["min_expedition"]:
            raise ValueError("Migration amount must be >= 2")

        sp = species[sid]
        if src not in sp["population"]:
            raise ValueError(f"{sid} has no population in {src}")

        available = float(sp["population"][src])
        cap = available * float(RULES["migration_cap_frac"])
        if amt > cap + 1e-9:
            raise ValueError(f"Migration exceeds 50% cap for {sid} in {src}")
        if available + 1e-9 < amt:
            raise ValueError("Not enough individuals to migrate")

        sp["population"][src] = available - amt
        if sp["population"][src] <= 1e-9:
            del sp["population"][src]

        sp["population"][dst] = float(sp["population"].get(dst, 0.0)) + amt
        touched_dst.add(dst)

    return touched_dst

def get_species_stats(save_path: str, species_id: str) -> dict:
    s = read_save(save_path)
    sp = s["player"]["species"][species_id]
    return {
        "species_id": species_id,
        "r": float(sp["stats"]["r"]),
        "b": {k: float(v) for k, v in sp["stats"]["b"].items()},
        "traits": _traits_counters(sp),
        "next_trait_cost": float(trait_cost(sp)),
        "population_by_cell": {k: float(v) for k, v in sp["population"].items()},
    }

def get_economy(save_path: str) -> dict:
    s = read_save(save_path)
    costs = {sid: float(trait_cost(sp)) for sid, sp in s["player"]["species"].items()}
    return {
        "evo_balance": float(s["player"]["evo"]),
        "passive_preview_next": float(s["state"].get("passive_preview", 0.0)),
        "trait_costs": costs,
        "fork_cost": float(fork_cost(len(s["player"]["species"]))),
    }

def get_cell_passport(save_path: str, cell_name: str) -> dict:
    s = read_save(save_path)
    return _make_cell_passport(s, cell_name, tick_resolved=s["state"]["tick"])

def build_hud(save_obj: dict) -> dict:
    tot_player = total_player_biomass(save_obj)
    species_count = len(save_obj["player"]["species"])

    # npc total
    tot_npc = 0.0
    for cell in (save_obj.get("world", {}).get("cells", {}) or {}).values():
        for npc in (cell.get("npc_species") or []):
            tot_npc += float(npc.get("pop", 0.0))

    denom = tot_player + tot_npc
    dominance = (tot_player / denom) if denom > 0 else 0.0

    pop_by_cell = {}
    for _sid, sp in save_obj["player"]["species"].items():
        for c, n in sp["population"].items():
            pop_by_cell[c] = pop_by_cell.get(c, 0.0) + float(n)

    return {
        "tick": int(save_obj["state"]["tick"]),
        "evo_balance": float(save_obj["player"]["evo"]),
        "passive_preview_next": float(save_obj["state"].get("passive_preview", 0.0)),
        "total_population": float(tot_player),
        "npc_total": float(tot_npc),
        "npc_total_population": float(tot_npc),
        "dominance": float(dominance),
        "dominance_ratio": float(dominance),
        "species_count": int(species_count),
        "player_pop_by_cell": {k: float(v) for k, v in sorted(pop_by_cell.items())},
    }


def build_economy(save_obj: dict) -> dict:
    costs = {sid: float(trait_cost(sp)) for sid, sp in save_obj["player"]["species"].items()}
    return {
        "evo_balance": float(save_obj["player"]["evo"]),
        "passive_preview_next": float(save_obj["state"].get("passive_preview", 0.0)),
        "trait_costs": costs,
        "fork_cost": float(fork_cost(len(save_obj["player"]["species"]))),
    }

def simulate_tick(save_path: str, action: dict=None, out_dir="/content"):
    """
    action schema (all optional):
    {
      "buy_trait": ("sp0","r") | ("sp0","b2"),
      "fork": ("sp0","ALL" or 6.0, "b2", "cell0"),
      "migrations": [("sp0","start","cell0",2.0), ...]
    }

    returns: (new_save_path, result_dict)
    result_dict: { hud, events, reveals, economy }
    """
    save_obj = read_save(save_path)
    action = action or {}
    species = save_obj["player"]["species"]

    # ---- EVO passive credit at START of tick (between-ticks canon) ----
    passive_now = RULES["evo_passive_k"] * math.sqrt(max(0.0, total_player_biomass(save_obj)))
    save_obj["player"]["evo"] = float(save_obj["player"]["evo"]) + float(passive_now)
    passive_credited = float(passive_now)

    # ---- Pre-tick: buy trait ----
    bt = action.get("buy_trait")
    if bt is not None:
        sid, stat = bt
        if sid not in species:
            raise ValueError("Unknown species for trait: " + str(sid))
        spec = species[sid]
        cost = trait_cost(spec)
        if float(save_obj["player"]["evo"]) + 1e-9 < cost:
            raise ValueError("Not enough EVO for trait")
        save_obj["player"]["evo"] = float(save_obj["player"]["evo"]) - float(cost)

        # traits are stored as a dict of counters, e.g. {"r": 2, "b3": 1}
        if ("traits" not in spec) or (not isinstance(spec["traits"], dict)):
            spec["traits"] = {}

        if stat == "r":
            spec["stats"]["r"] = float(spec["stats"]["r"]) + 0.1
            spec["traits"]["r"] = int(spec["traits"].get("r", 0)) + 1
        elif stat in ("b1", "b2", "b3"):
            key = "R1" if stat == "b1" else ("R2" if stat == "b2" else "R3")
            spec["stats"]["b"][key] = float(spec["stats"]["b"][key]) + 0.1
            spec["traits"][stat] = int(spec["traits"].get(stat, 0)) + 1
        else:
            raise ValueError("Trait stat must be r|b1|b2|b3")

    # ---- Pre-tick: fork ----
    fk = action.get("fork")
    if fk is not None:
        parent_id, split_amt, starter, source_cell = fk
        if parent_id not in species:
            raise ValueError("Unknown parent species: " + str(parent_id))

        cost = fork_cost(len(species))
        if float(save_obj["player"]["evo"]) + 1e-9 < cost:
            raise ValueError("Not enough EVO for fork")

        parent = species[parent_id]
        if source_cell not in parent["population"]:
            raise ValueError("Parent has no population in " + str(source_cell))

        take = float(parent["population"][source_cell]) if (isinstance(split_amt, str) and split_amt.upper() == "ALL") else float(split_amt)
        if take < RULES["min_expedition"]:
            raise ValueError("Fork split must be >= 2")
        if float(parent["population"][source_cell]) + 1e-9 < take:
            raise ValueError("Not enough individuals in source cell to fork")

        parent["population"][source_cell] = float(parent["population"][source_cell]) - take
        if parent["population"][source_cell] <= 1e-9:
            del parent["population"][source_cell]

        new_id = "sp" + str(len(species))
        new_stats = {"r": float(parent["stats"]["r"]), "b": dict(parent["stats"]["b"])}
        new_traits = {"r": 0, "b1": 0, "b2": 0, "b3": 0}

        if starter == "r":
            new_stats["r"] += 0.1
            new_traits["r"] = 1
        elif starter in ("b1", "b2", "b3"):
            key = "R1" if starter == "b1" else ("R2" if starter == "b2" else "R3")
            new_stats["b"][key] = float(new_stats["b"][key]) + 0.1
            new_traits[starter] = 1
        else:
            raise ValueError("Fork starter must be r|b1|b2|b3")

        species[new_id] = {"stats": new_stats, "traits": new_traits, "population": {source_cell: take}}
        save_obj["player"]["evo"] = float(save_obj["player"]["evo"]) - float(cost)

    # ---- 1) Migrations ----
    touched_dst = apply_migrations(save_obj, action.get("migrations"))

    # ---- 2) Births ----
    pre_map = {sid: births(sp["population"], float(sp["stats"]["r"])) for sid, sp in species.items()}

    # ---- 3) Events ----
    happened, event_info = event_roll(save_obj)
    events_out = []
    if happened and event_info and event_info.get("type") != "noop":
        if event_info["type"] == "fluctuation":
            save_obj["state"].setdefault("active_effects", [])
            save_obj["state"]["active_effects"].append(event_info["eff"])
            events_out.append({"type": "fluctuation", "cell": event_info["cell"], "res": event_info["res"],
                               "mult": event_info["mult"], "ttl": event_info["ttl"]})
        elif event_info["type"] == "invasive":
            cell = event_info["cell"]
            save_obj["world"]["cells"][cell].setdefault("npc_species", [])
            save_obj["world"]["cells"][cell]["npc_species"].append(event_info["npc"])
            events_out.append({"type": "invasive", "cell": event_info["cell"], "res": event_info["res"],
                               "pop": event_info["npc"]["pop"]})

    active_effects = _apply_active_effects(save_obj)

    # ---- 4-5) Allocation + starvation; capture breakdown ----
    last_breakdown = {}
    new_pops = {sid: {} for sid in species.keys()}

    for cell_name, cell in save_obj["world"]["cells"].items():
        contestants = []

        # player species present
        for sid in species.keys():
            npre = float(pre_map[sid].get(cell_name, 0.0))
            if npre > 0.0:
                contestants.append({"id": sid, "kind": "player", "n_pre": npre, "b": species[sid]["stats"]["b"]})

        # NPC present
        npcs = cell.get("npc_species", [])
        for i, npc in enumerate(npcs):
            npc_pre = float(npc["pop"]) * (1.0 + float(npc["r"]))
            contestants.append({"id": f"npc{i}", "kind": "npc", "n_pre": npc_pre, "b": npc["b"]})

        if not contestants:
            continue

        eff_res = _effective_resources(cell, active_effects.get(cell_name, {}))
        cell_capacity_total = float(sum(float(v) for v in eff_res.values()))
        food = alloc_food_for_cell(eff_res, contestants)

        rows = []
        for c in contestants:
            cid = c["id"]
            npre = float(c["n_pre"])
            got = float(food.get(cid, 0.0))
            surv = min(npre, got)

            if c["kind"] == "player":
                if surv > 1e-9:
                    new_pops[cid][cell_name] = float(surv)
            else:
                idx = int(cid.replace("npc", ""))
                npcs[idx]["pop"] = float(surv)

            food_eaten = min(npre, got)
            share_of_capacity = (food_eaten / max(1e-12, cell_capacity_total))
            rows.append({
                "id": cid,
                "kind": c["kind"],
                "need": float(npre),
                "food_allocated": float(got),
                "food_eaten": float(food_eaten),
                "survivors": float(surv),
                "cell_capacity_total": float(cell_capacity_total),
                "share_of_capacity": float(share_of_capacity),
            })

        last_breakdown[cell_name] = {
            "resources_effective": {k: float(v) for k, v in eff_res.items()},
            "rows": rows
        }

    # commit player populations
    for sid in species.keys():
        species[sid]["population"] = new_pops[sid]

    save_obj["state"]["last_tick_breakdown"] = last_breakdown

    # For UI: remember share-of-capacity by id per cell for the last processed tick
    last_share = {}
    for _cell, bd in last_breakdown.items():
        for row in bd.get("rows", []):
            rid = row.get("id")
            if not rid:
                continue
            last_share.setdefault(rid, {})[_cell] = float(row.get("share_of_capacity", 0.0))
    save_obj["state"]["last_share_of_capacity"] = last_share
    # Passive preview for NEXT tick (after this tick resolves)
    save_obj["state"]["passive_preview"] = float(RULES["evo_passive_k"] * math.sqrt(max(0.0, total_player_biomass(save_obj))))


    # ---- 6) Tick down effects ----
    _tick_down_effects(save_obj)

    # ---- 7) Economy: scouting + compute passive next ----
    evo_explore = 0.0
    newly_scouted = []
    # Backward/forward compatibility: older saves may not have this key.
    if "scouted_cells" not in save_obj["state"]:
        save_obj["state"]["scouted_cells"] = list(save_obj.get("player", {}).get("explored", []))
    for dst in sorted(list(touched_dst)):
        if dst not in save_obj["state"]["scouted_cells"]:
            save_obj["state"]["scouted_cells"].append(dst)
            evo_explore += float(RULES["explore_reward"])
            newly_scouted.append(dst)

    if evo_explore:
        save_obj["player"]["evo"] = float(save_obj["player"]["evo"]) + evo_explore

    B_total = total_player_biomass(save_obj)
    passive_next = RULES["evo_passive_k"] * math.sqrt(B_total) if B_total > 0 else 0.0
    save_obj["state"]["passive_preview"] = float(passive_next)

    # advance tick number
    save_obj["state"]["tick"] = int(save_obj["state"]["tick"]) + 1
    resolved_tick = int(save_obj["state"]["tick"])

    # ---- 8) Reveals AFTER tick resolves ----
    reveals = []
    for c in newly_scouted:
        reveals.append(_make_cell_passport(save_obj, c, tick_resolved=resolved_tick))

    # Cache last-tick UI outputs so a UI can be derived from the save alone.
    save_obj["state"]["last_events"] = list(events_out)
    save_obj["state"]["last_reveals"] = list(reveals)

    save_obj["state"]["history"].append({
        "tick": resolved_tick,
        "action": action,
        "events": events_out,
        "evo_explore": evo_explore,
        "passive_credited": passive_credited,
        "passive_due_next": passive_next,
        "newly_scouted": newly_scouted,
    })

    # ---- 8) Update cached HUD/economy & pop history for UI ----
    if "pop_history" not in save_obj["state"]:
        save_obj["state"]["pop_history"] = {}
    for sid, sp in save_obj["player"]["species"].items():
        for cell_id, pop in sp.get("population", {}).items():
            save_obj["state"]["pop_history"].setdefault(sid, {}).setdefault(cell_id, []).append(float(pop))
            # cap history length
            if len(save_obj["state"]["pop_history"][sid][cell_id]) > 30:
                save_obj["state"]["pop_history"][sid][cell_id] = save_obj["state"]["pop_history"][sid][cell_id][-30:]

    # total (player) pop history per cell for map trend
    if "pop_history_total" not in save_obj["state"]:
        save_obj["state"]["pop_history_total"] = {}
    tot_by_cell = {}
    for _sid, _sp in save_obj["player"]["species"].items():
        for _c, _n in _sp.get("population", {}).items():
            tot_by_cell[_c] = tot_by_cell.get(_c, 0.0) + float(_n)
    for _c, _n in tot_by_cell.items():
        save_obj["state"]["pop_history_total"].setdefault(_c, []).append(float(_n))
        if len(save_obj["state"]["pop_history_total"][_c]) > 30:
            save_obj["state"]["pop_history_total"][_c] = save_obj["state"]["pop_history_total"][_c][-30:]

    save_obj["state"]["hud"] = build_hud(save_obj)
    save_obj["state"]["economy"] = build_economy(save_obj)

    # save
    if save_obj["meta"]["save_id"].startswith("seed_"):
        save_obj["meta"]["save_id"] = "run_" + save_obj["meta"]["save_id"] + "_" + str(int(time.time()))
    out_path = os.path.join(out_dir, save_obj["meta"]["save_id"] + ".json")
    write_save(out_path, save_obj)

    result = {
        "hud": build_hud(save_obj),
        "events": events_out,
        "reveals": reveals,
        "economy": get_economy(out_path),
    }
    return out_path, result

# ---------------- UI helper API (read-only) ----------------
def get_hud(save_path: str) -> dict:
    s = read_save(save_path)
    hud = s.get('state', {}).get('hud')
    if isinstance(hud, dict) and hud.get('tick') is not None:
        # If cache predates dominance fields, rebuild.
        if ('dominance' in hud) and ('npc_total' in hud):
            return hud
    return build_hud(s)

def get_economy_cached(save_path: str) -> dict:
    s = read_save(save_path)
    econ = s.get("state", {}).get("economy")
    if isinstance(econ, dict) and econ.get("trait_costs") is not None:
        return econ
    return build_economy(s)

def get_species_list(save_path: str) -> list:
    s = read_save(save_path)
    econ = get_economy_cached(save_path)
    out = []
    for sid, sp in s["player"]["species"].items():
        out.append({
            "species_id": sid,
            "stats": {"r": float(sp["stats"]["r"]), "b": {k: float(v) for k, v in sp["stats"]["b"].items()}},
            "traits": _traits_counters(sp),
            "trait_cost": float(econ.get("trait_costs", {}).get(sid, trait_cost(sp))),
            "fork_cost": float(econ.get("fork_cost", fork_cost(len(s["player"]["species"])))),
        })
    return out

def get_species_cells(save_path: str, species_id: str) -> dict:
    """Per-cell data for one species, derived from last tick state.

    trend is based on the SIGN of population delta between last two ticks:
    - up / down / flat
    """
    s = read_save(save_path)
    sp = s.get("player", {}).get("species", {}).get(species_id)
    if not sp:
        return {}

    pop_by_cell = {k: float(v) for k, v in (sp.get("population", {}) or {}).items() if float(v) > RULES.get("extinction_eps", 0.0)}
    shares = s.get("state", {}).get("last_share_of_capacity", {}).get(species_id, {})
    hist = s.get("state", {}).get("pop_history", {}).get(species_id, {})

    out = {}
    for cell, pop in pop_by_cell.items():
        share = float(shares.get(cell, 0.0))
        h = list(hist.get(cell, []))
        trend = "flat"
        if len(h) >= 2:
            d = float(h[-1]) - float(h[-2])
            if d > 1e-9:
                trend = "up"
            elif d < -1e-9:
                trend = "down"
        out[cell] = {
            "population": float(pop),
            "consumption_share": share,
            "trend": trend,
        }
    return out

def get_map_state(save_path: str, selected_cell: str | None = None) -> dict:
    """Map overview for UI.

    Includes fog-of-war flags, neighbor list, and (if available) last-tick fill + total trend.
    """
    s = read_save(save_path)
    world_cells = s.get("world", {}).get("cells", {}) or {}
    scouted = set(s.get("state", {}).get("scouted_cells", []) or [])

    # player presence per cell
    player_cells = set()
    for _sid, _sp in s.get("player", {}).get("species", {}).items():
        for c in (_sp.get("population", {}) or {}).keys():
            player_cells.add(c)

    # fill segments from last_share_of_capacity
    last_share = s.get("state", {}).get("last_share_of_capacity", {}) or {}

    def _fill_for_cell(cid: str):
        segs = []
        used = 0.0
        # aggregate npc shares
        npc_share = 0.0
        for rid, mp in last_share.items():
            if isinstance(rid, str) and rid.startswith("npc"):
                npc_share += float((mp or {}).get(cid, 0.0) or 0.0)
        if npc_share > 1e-9:
            segs.append({"id": "npc", "share": float(npc_share)})
            used += npc_share
        # player species shares
        for sid, mp in last_share.items():
            if not isinstance(sid, str) or not sid.startswith("sp"):
                continue
            sh = float((mp or {}).get(cid, 0.0) or 0.0)
            if sh > 1e-9:
                segs.append({"id": sid, "share": float(sh)})
                used += sh
        unused = max(0.0, 1.0 - used)
        if unused > 1e-9:
            segs.append({"id": "unused", "share": float(unused)})
        # normalize if over 1
        total = sum(float(x.get("share", 0.0)) for x in segs)
        if total > 1.0 + 1e-6:
            segs = [{"id": x["id"], "share": float(x.get("share", 0.0))/total} for x in segs]
        return {"segments": segs}

    # total trend based on player pop (sum of player species) history
    hist_total = s.get("state", {}).get("pop_history_total", {}) or {}
    def _trend_total(cid: str) -> str:
        h = list(hist_total.get(cid, []) or [])
        if len(h) >= 2:
            d = float(h[-1]) - float(h[-2])
            if d > 1e-9:
                return "up"
            if d < -1e-9:
                return "down"
        return "flat"

    cells = {}
    for cid, cell in world_cells.items():
        is_scouted = cid in scouted
        cells[cid] = {
            "has_player": cid in player_cells,
            "has_npc": bool(cell.get("npc_species")),
            "is_scouted": bool(is_scouted),
            "neighbors": list(cell.get("neighbors", [])),
        }
        if is_scouted:
            cells[cid]["fill"] = _fill_for_cell(cid)
            cells[cid]["trend_total"] = _trend_total(cid)

    return {"selected_cell": selected_cell, "cells": cells}

def get_cell_inspector(save_path: str, cell_id: str) -> dict:
    s = read_save(save_path)
    scouted = set(s.get('state', {}).get('scouted_cells', []) or [])
    has_player = any(cell_id in (_sp.get('population', {}) or {}) for _sp in s.get('player', {}).get('species', {}).values())
    if (cell_id not in scouted) and (not has_player):
        # fog-of-war: only show minimal metadata
        cell = s.get('world', {}).get('cells', {}).get(cell_id, {}) or {}
        return {
            'cell_id': cell_id,
            'is_scouted': False,
            'neighbors': list(cell.get('neighbors', [])),
            'resources': None,
            'populations': None,
            'passport': None,
        }
    passport = _make_cell_passport(s, cell_id, tick_resolved=s["state"]["tick"])
    # Build populations list (player + npc) with absolute numbers
    pops = []
    # player
    for sid, sp in s["player"]["species"].items():
        if cell_id in sp.get("population", {}):
            pops.append({"id": sid, "kind": "player", "population": float(sp["population"][cell_id])})
    # npc
    for i, npc in enumerate(s["world"]["cells"][cell_id].get("npc_species", [])):
        pops.append({"id": f"npc{i}", "kind": "npc", "population": float(npc.get("pop", 0.0))})
    # sort desc
    pops = sorted(pops, key=lambda x: -x["population"])
    return {
        "cell_id": cell_id,
        "resources": passport.get("resources", {}),
        "neighbors": list(s["world"]["cells"][cell_id].get("neighbors", passport.get("neighbors", []))),
        "populations": pops,
        "passport": passport,
    }

def get_cell_consumption(save_path: str, cell_id: str) -> dict:
    s = read_save(save_path)
    bd = s.get("state", {}).get("last_tick_breakdown", {}).get(cell_id, {})
    rows = []
    cap_total = 0.0
    if bd:
        # all rows already have share_of_capacity and cell_capacity_total
        for row in bd.get("rows", []):
            cap_total = float(row.get("cell_capacity_total", cap_total))
            rows.append({
                "id": row.get("id"),
                "kind": row.get("kind"),
                "share": float(row.get("share_of_capacity", 0.0)),
                "food_eaten": float(row.get("food_eaten", 0.0)),
                "survivors": float(row.get("survivors", 0.0)),
            })
    return {"cell_id": cell_id, "cell_capacity_total": cap_total, "rows": rows, "resources_effective": bd.get("resources_effective", {})}


def get_ui_snapshot(save_path: str, selected_cell: str | None = None) -> dict:
    """Single-call UI snapshot.

    Intended for Streamlit: one call per render.
    Includes: HUD, economy, species list, per-species per-cell rows, map overview,
    last tick events/reveals, and optional selected cell inspector/passport/consumption.
    """
    hud = get_hud(save_path)
    economy = get_economy_cached(save_path)
    species_list = get_species_list(save_path)

    species_cells = {}
    for sp in species_list:
        sid = sp.get("species_id")
        if sid:
            species_cells[sid] = get_species_cells(save_path, sid)

    # --- Active effects (global, always visible) ---
    s_full = read_save(save_path)
    active_effects_global = []
    for eff in (s_full.get("state", {}).get("active_effects", []) or []):
        try:
            active_effects_global.append({
                "type": str(eff.get("type", "fluctuation")),
                "cell": str(eff.get("cell")),
                "res": str(eff.get("res")),
                "mult": float(eff.get("mult")),
                "ttl": int(eff.get("ttl")),
            })
        except Exception:
            continue

    snapshot = {
        "hud": hud,
        "economy": economy,
        "species": species_list,
        "species_cells": species_cells,
        "map": get_map_state(save_path, selected_cell=selected_cell),
        "last_events": read_save(save_path).get("state", {}).get("last_events", []),
        "active_effects": active_effects_global,
        "last_reveals": read_save(save_path).get("state", {}).get("last_reveals", []),
        "selected_cell": None,
    }

    if selected_cell is not None:
        snapshot["selected_cell"] = {
            "cell_id": selected_cell,
            "inspector": get_cell_inspector(save_path, selected_cell),
            "passport": get_cell_passport(save_path, selected_cell),
            "consumption": get_cell_consumption(save_path, selected_cell),
        }

    return snapshot
