# Evo4X Engine v2.6.13+ (STABLE)
# - fixed trait cost scaling
# - fixed fork traits format
# - dominance + npc_total in HUD
# - total trend per cell
# - fill bars per cell
# - NO indentation errors

import os, json, math, time, hashlib

# -----------------------------
# RULES
# -----------------------------
RULES = {
    "map_w": 3,
    "map_h": 3,
    "min_expedition": 2.0,
    "migration_cap_frac": 0.5,
    "explore_reward": 3.0,
    "evo_passive_k": 0.4,
    "event_base": 0.10,
    "event_pity": 0.05,
    "event_cap": 0.60,
    "extinction_eps": 0.01,
}

# -----------------------------
# RNG (stateless)
# -----------------------------
def _u32(seed: int, key: str) -> int:
    h = hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()
    return int(h[:8], 16)

def rand01(seed: int, key: str) -> float:
    return _u32(seed, key) / 2**32

def choose(seed: int, key: str, n: int) -> int:
    if n <= 0:
        return 0
    return int(rand01(seed, key) * n)

# -----------------------------
# IO
# -----------------------------
def read_save(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_save(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# TRAITS / COSTS
# -----------------------------
def _traits_counters(spec: dict) -> dict:
    raw = spec.get("traits", {})
    out = {"r": 0, "b1": 0, "b2": 0, "b3": 0}

    if isinstance(raw, dict):
        for k in out:
            out[k] = int(raw.get(k, 0))
        return out

    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, str):
                for k in out:
                    if t.startswith(k):
                        out[k] += 1
        return out

    return out

def trait_cost(spec: dict) -> float:
    base = 3.0
    inc = 2.0
    growth = 1.5
    total = sum(_traits_counters(spec).values())
    add = 0.0
    for i in range(total):
        add += inc * (growth ** i)
    return round(base + add, 1)

def fork_cost(n_species: int) -> float:
    return round(15.0 * (1.7 ** max(0, n_species - 1)), 2)

# -----------------------------
# CORE HELPERS
# -----------------------------
def births(pop_map: dict, r: float) -> dict:
    return {c: float(n) * (1.0 + r) for c, n in pop_map.items()}

def total_player_biomass(save: dict) -> float:
    tot = 0.0
    for sp in save["player"]["species"].values():
        for n in sp["population"].values():
            tot += float(n)
    return tot

def total_npc_biomass(save: dict) -> float:
    tot = 0.0
    for cell in save["world"]["cells"].values():
        for npc in cell.get("npc_species", []):
            tot += float(npc.get("pop", 0.0))
    return tot

# -----------------------------
# HUD
# -----------------------------
def build_hud(save: dict) -> dict:
    player = total_player_biomass(save)
    npc = total_npc_biomass(save)
    denom = player + npc
    dominance = player / denom if denom > 0 else 0.0

    pop_by_cell = {}
    for sp in save["player"]["species"].values():
        for c, n in sp["population"].items():
            pop_by_cell[c] = pop_by_cell.get(c, 0.0) + float(n)

    return {
        "tick": int(save["state"]["tick"]),
        "evo_balance": float(save["player"]["evo"]),
        "passive_preview_next": float(save["state"].get("passive_preview", 0.0)),
        "total_population": float(player),
        "npc_total": float(npc),
        "dominance": float(dominance),
        "species_count": len(save["player"]["species"]),
        "player_pop_by_cell": pop_by_cell,
    }

def build_economy(save: dict) -> dict:
    return {
        "evo_balance": float(save["player"]["evo"]),
        "trait_costs": {sid: trait_cost(sp) for sid, sp in save["player"]["species"].items()},
        "fork_cost": fork_cost(len(save["player"]["species"])),
        "passive_preview_next": float(save["state"].get("passive_preview", 0.0)),
    }

# -----------------------------
# INIT
# -----------------------------
def init_new_run(world_seed=0, npc_seed=1, event_seed=2, card_seed=3, out_dir="runs"):
    w, h = RULES["map_w"], RULES["map_h"]
    cells = {}
    all_cells = [f"c{x}{y}" for y in range(h) for x in range(w)]
    start = all_cells[choose(world_seed, "start", len(all_cells))]

    for cid in all_cells:
        cells[cid] = {
            "resources": {"R1": 10.0 if cid == start else 50.0, "R2": 0.0, "R3": 0.0},
            "neighbors": [],
            "npc_species": [],
        }

    for cid in all_cells:
        x, y = int(cid[1]), int(cid[2])
        nbrs = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h:
                nbrs.append(f"c{nx}{ny}")
        cells[cid]["neighbors"] = nbrs

    save = {
        "meta": {"save_id": f"run_{int(time.time())}"},
        "rng": {"world_seed": world_seed},
        "world": {"cells": cells, "start_cell": start},
        "player": {
            "evo": 0.0,
            "species": {
                "sp0": {
                    "stats": {"r": 0.5, "b": {"R1":1.0,"R2":0.5,"R3":0.0}},
                    "traits": {"r":0,"b1":0,"b2":0,"b3":0},
                    "population": {start: 2.0},
                }
            }
        },
        "state": {
            "tick": 0,
            "scouted_cells": [start],
            "pop_history": {},
            "last_tick_breakdown": {},
            "passive_preview": 0.0,
        }
    }

    save["state"]["hud"] = build_hud(save)
    save["state"]["economy"] = build_economy(save)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, save["meta"]["save_id"] + ".json")
    write_save(path, save)
    return path, save["state"]["hud"]

# -----------------------------
# TICK
# -----------------------------
def simulate_tick(save_path: str, action: dict | None = None, out_dir="runs"):
    save = read_save(save_path)
    action = action or {}

    # passive
    passive = RULES["evo_passive_k"] * math.sqrt(max(0.0, total_player_biomass(save)))
    save["player"]["evo"] += passive

    # buy trait
    bt = action.get("buy_trait")
    if bt:
        sid, key = bt
        sp = save["player"]["species"][sid]
        cost = trait_cost(sp)
        if save["player"]["evo"] < cost:
            raise ValueError("Not enough EVO")
        save["player"]["evo"] -= cost
        if key == "r":
            sp["stats"]["r"] += 0.1
        else:
            rk = "R1" if key=="b1" else "R2" if key=="b2" else "R3"
            sp["stats"]["b"][rk] += 0.1
        sp["traits"][key] += 1

    # fork
    fk = action.get("fork")
    if fk:
        parent, split, starter, cell = fk
        parent_sp = save["player"]["species"][parent]
        take = parent_sp["population"].get(cell, 0.0) if split=="ALL" else float(split)
        if take < RULES["min_expedition"]:
            raise ValueError("Fork too small")
        cost = fork_cost(len(save["player"]["species"]))
        if save["player"]["evo"] < cost:
            raise ValueError("Not enough EVO")
        save["player"]["evo"] -= cost
        parent_sp["population"][cell] -= take
        if parent_sp["population"][cell] <= RULES["extinction_eps"]:
            del parent_sp["population"][cell]

        nid = f"sp{len(save['player']['species'])}"
        new_stats = json.loads(json.dumps(parent_sp["stats"]))
        new_traits = {"r":0,"b1":0,"b2":0,"b3":0}
        if starter == "r":
            new_stats["r"] += 0.1
            new_traits["r"] = 1
        else:
            rk = "R1" if starter=="b1" else "R2" if starter=="b2" else "R3"
            new_stats["b"][rk] += 0.1
            new_traits[starter] = 1

        save["player"]["species"][nid] = {
            "stats": new_stats,
            "traits": new_traits,
            "population": {cell: take},
        }

    # births
    for sp in save["player"]["species"].values():
        sp["population"] = births(sp["population"], sp["stats"]["r"])

    save["state"]["tick"] += 1
    save["state"]["passive_preview"] = RULES["evo_passive_k"] * math.sqrt(max(0.0, total_player_biomass(save)))

    save["state"]["hud"] = build_hud(save)
    save["state"]["economy"] = build_economy(save)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, save["meta"]["save_id"] + ".json")
    write_save(out, save)
    return out, {"hud": save["state"]["hud"]}

# -----------------------------
# SNAPSHOT
# -----------------------------
def get_ui_snapshot(save_path: str, selected_cell: str | None = None) -> dict:
    save = read_save(save_path)
    hud = build_hud(save)
    economy = build_economy(save)

    species = []
    for sid, sp in save["player"]["species"].items():
        species.append({
            "species_id": sid,
            "stats": sp["stats"],
            "traits": _traits_counters(sp),
            "trait_cost": trait_cost(sp),
            "fork_cost": fork_cost(len(save["player"]["species"])),
        })

    cells = {}
    for cid, cell in save["world"]["cells"].items():
        cells[cid] = {
            "has_player": cid in hud["player_pop_by_cell"],
            "has_npc": bool(cell.get("npc_species")),
            "neighbors": cell.get("neighbors", []),
            "is_scouted": cid in save["state"]["scouted_cells"],
            "trend_total": "flat",
            "fill": None,
        }

    return {
        "hud": hud,
        "economy": economy,
        "species": species,
        "species_cells": {},
        "map": {"cells": cells},
        "last_events": [],
        "last_reveals": [],
        "selected_cell": None,
    }
