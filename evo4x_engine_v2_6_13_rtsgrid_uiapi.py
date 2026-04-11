# Evo4X Engine v2.6 (Streamlit-friendly)
# - Deterministic RNG via sha256 (stateless)
# - simulate_tick returns dicts: HUD + reveals + events + economy
# - Reveal cell passport AFTER the tick resolves (state at that moment)
# - EVO passive credited at START of tick from stored passive_due
# - Migrations are species-aware

import os, json, math, time, hashlib, pathlib

_DATA_DIR = pathlib.Path(__file__).parent / "data"

def _load_json(name: str) -> dict:
    p = _DATA_DIR / name
    with open(p, encoding="utf-8") as f:
        return json.load(f)

RULES = {
    'extinction_eps': 1.0,
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

    "npc_b3_cap": 0.30,
    "npc_r_min": 0.10,
    "npc_r_max": 0.60,
}

# ---------- Savanna world data ----------
# Loaded from data/species.json and data/archetypes.json — edit those files to tune balance.

SAVANNA_SPECIES: dict = _load_json("species.json")

SAVANNA_RESOURCE_UI = {
    "R1": {"ui_name": "Grass",  "emoji": "🌱"},
    "R2": {"ui_name": "Leaves", "emoji": "🌿"},
    "R3": {"ui_name": "Roots",  "emoji": "🥔"},
}

SAVANNA_PLAYER_ICONS = {"sp0": "🧬", "sp1": "🐾", "sp2": "🦴", "sp3": "🦠"}

SAVANNA_RICHNESS = {"poor": 30, "normal": 60, "rich": 90}

SAVANNA_ARCHETYPES: dict = _load_json("archetypes.json")

# NPC initial pop by pack size (1/2/3) and richness tier
SAVANNA_NPC_POP = {
    1: {"poor":  8, "normal": 14, "rich": 22},
    2: {"poor":  6, "normal": 10, "rich": 16},
    3: {"poor":  5, "normal":  8, "rich": 12},
}

# 12 layout templates (archetype per non-start cell)
SAVANNA_TEMPLATES = [
    {"c00":"dense_grove",  "c01":"woodland",        "c02":"rootland",
     "c10":"open_savanna", "c12":"root_patch",       "c20":"grassland",    "c21":"diverse_savanna","c22":"grassland"},
    {"c00":"rootland",     "c01":"open_savanna",     "c02":"dense_grove",
     "c10":"root_patch",   "c12":"woodland",          "c20":"grassland",    "c21":"root_mosaic",    "c22":"grassland"},
    {"c00":"grassland",    "c01":"woodland",          "c02":"grassland",
     "c10":"open_savanna", "c12":"root_patch",        "c20":"rootland",     "c21":"leaf_mosaic",    "c22":"dense_grove"},
    {"c00":"dense_grove",  "c01":"root_patch",        "c02":"grassland",
     "c10":"woodland",     "c12":"open_savanna",      "c20":"rootland",     "c21":"grassland",      "c22":"diverse_savanna"},
    {"c00":"grassland",    "c01":"rootland",          "c02":"dense_grove",
     "c10":"open_savanna", "c12":"woodland",          "c20":"diverse_savanna","c21":"grassland",    "c22":"root_patch"},
    {"c00":"root_patch",   "c01":"woodland",          "c02":"grassland",
     "c10":"rootland",     "c12":"dense_grove",       "c20":"grassland",    "c21":"open_savanna",   "c22":"leaf_mosaic"},
    {"c00":"dense_grove",  "c01":"grassland",         "c02":"root_patch",
     "c10":"woodland",     "c12":"rootland",          "c20":"diverse_savanna","c21":"open_savanna", "c22":"grassland"},
    {"c00":"grassland",    "c01":"dense_grove",       "c02":"grassland",
     "c10":"root_patch",   "c12":"woodland",          "c20":"rootland",     "c21":"diverse_savanna","c22":"open_savanna"},
    {"c00":"rootland",     "c01":"grassland",         "c02":"dense_grove",
     "c10":"woodland",     "c12":"root_patch",        "c20":"open_savanna", "c21":"leaf_mosaic",    "c22":"grassland"},
    {"c00":"grassland",    "c01":"root_patch",        "c02":"rootland",
     "c10":"dense_grove",  "c12":"woodland",          "c20":"open_savanna", "c21":"diverse_savanna","c22":"grassland"},
    {"c00":"dense_grove",  "c01":"leaf_mosaic",       "c02":"grassland",
     "c10":"open_savanna", "c12":"root_patch",        "c20":"grassland",    "c21":"woodland",       "c22":"rootland"},
    {"c00":"grassland",    "c01":"diverse_savanna",   "c02":"grassland",
     "c10":"rootland",     "c12":"open_savanna",      "c20":"dense_grove",  "c21":"root_patch",     "c22":"woodland"},
]

# Slot options for the 4 neighbors of c11 (spec: N=c01, W=c10, E=c12, S=c21)
SAVANNA_SLOT_OPTIONS = {
    "safe_training": [
        {"archetype": "grassland",        "richness": "normal", "pack": []},
        {"archetype": "grassland",        "richness": "normal", "pack": ["gazelle"]},
    ],
    "leaf_offspec": [
        {"archetype": "open_savanna",     "richness": "normal", "pack": ["giraffe"]},
        {"archetype": "open_savanna",     "richness": "normal", "pack": ["giraffe", "impala"]},
        {"archetype": "woodland",         "richness": "normal", "pack": ["giraffe"]},
        {"archetype": "woodland",         "richness": "normal", "pack": ["giraffe", "impala"]},
    ],
    "root_offspec": [
        {"archetype": "root_patch",       "richness": "normal", "pack": ["mole_rat"]},
        {"archetype": "rootland",         "richness": "normal", "pack": ["mole_rat"]},
    ],
    "wildcard": [
        {"archetype": "open_savanna",     "richness": "normal", "pack": ["giraffe"]},
        {"archetype": "open_savanna",     "richness": "normal", "pack": ["giraffe", "gazelle"]},
        {"archetype": "root_patch",       "richness": "normal", "pack": ["mole_rat"]},
        {"archetype": "root_patch",       "richness": "normal", "pack": ["warthog"]},
        {"archetype": "woodland",         "richness": "normal", "pack": ["giraffe"]},
        {"archetype": "diverse_savanna",  "richness": "poor",   "pack": ["giraffe", "mole_rat", "gazelle"]},
    ],
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
    # Uniform pick in [0, n): rand01 ∈ [0,1) → int(rand01*n) ∈ {0,..,n-1}.
    return min(n - 1, int(rand01(seed_int, key) * n))

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
            "species_type": npc.get("species_type"),
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
        "archetype": cell.get("archetype"),
        "richness":  cell.get("richness"),
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

    # choose occupied cell (any player presence)
    occ = set()
    for sp in save_obj["player"]["species"].values():
        for c in sp["population"].keys():
            occ.add(c)
    occ = sorted(list(occ))
    if not occ:
        return True, {"type": "noop"}

    cell = occ[choose_index(save_obj["rng"]["event_seed"], f"tick:{t}:cell", len(occ))]
    cell_res = save_obj["world"]["cells"].get(cell, {}).get("resources", {})
    res_list = [r for r in ("R1", "R2", "R3") if float(cell_res.get(r, 0.0)) > 0]
    if not res_list:
        return True, {"type": "noop"}
    res = res_list[choose_index(save_obj["rng"]["event_seed"], f"tick:{t}:res:{cell}", len(res_list))]

    # Only event type: fluctuation (invasive removed)
    mult = lerp(RULES["event_fluct_mult_min"], RULES["event_fluct_mult_max"],
                rand01(save_obj["rng"]["event_seed"], f"tick:{t}:mult:{cell}:{res}"))
    ttl = int(lerp(RULES["event_fluct_ttl_min"], RULES["event_fluct_ttl_max"] + 0.9999,
                   rand01(save_obj["rng"]["event_seed"], f"tick:{t}:ttl:{cell}:{res}")))
    eff = {"type": "fluctuation", "cell": cell, "res": res, "mult": float(mult), "ttl": ttl}
    return True, {"type": "fluctuation", "eff": eff, "cell": cell, "res": res, "mult": float(mult), "ttl": ttl}

# ---------- World generation ----------

# --- Savanna worldgen helpers ---

def _savanna_resources(archetype: str, richness: str) -> dict:
    """Integer R1/R2/R3 from archetype ratio and richness total. Sum always equals tier total."""
    ratio = SAVANNA_ARCHETYPES[archetype]["ratio"]
    total = SAVANNA_RICHNESS[richness]
    r1 = round(total * ratio["R1"] / 100)
    r2 = round(total * ratio["R2"] / 100)
    r3 = round(total * ratio["R3"] / 100)
    diff = total - (r1 + r2 + r3)
    if diff != 0:
        dom = max(ratio, key=lambda k: ratio[k])
        if dom == "R1":   r1 += diff
        elif dom == "R2": r2 += diff
        else:             r3 += diff
    return {"R1": float(r1), "R2": float(r2), "R3": float(r3)}


def _savanna_permute(seed_int: int, key: str, lst: list) -> list:
    """Fisher-Yates shuffle with stateless RNG."""
    lst = list(lst)
    for i in range(len(lst) - 1, 0, -1):
        j = choose_index(seed_int, f"{key}:{i}", i + 1)
        lst[i], lst[j] = lst[j], lst[i]
    return lst


def _rotate_cid(cid: str, n90: int) -> str:
    """Rotate cell id n90×90° CW around center of 3×3 grid. Formula: (x,y)->(2-y,x)."""
    x, y = int(cid[1]), int(cid[2])
    for _ in range(n90 % 4):
        x, y = 2 - y, x
    return f"c{x}{y}"


def _reflect_cid(cid: str) -> str:
    """Horizontal reflection: (x,y)->(2-x,y)."""
    return f"c{2 - int(cid[1])}{cid[2]}"


def _is_contested(pack: list) -> bool:
    return "buffalo" in pack or "elephant" in pack or len(pack) >= 3


def _savanna_npc_entry(species_type: str, pop: float) -> dict:
    sp = SAVANNA_SPECIES[species_type]
    return {
        "pop": float(pop),
        "r": float(sp["stats"]["r"]),
        "b": dict(sp["stats"]["b"]),
        "origin": "seeded",
        "species_type": species_type,
    }


def _build_npc_pack(pack: list, richness: str) -> list:
    if not pack:
        return []
    n = min(len(pack), 3)
    pop_val = float(SAVANNA_NPC_POP[n][richness])
    return [_savanna_npc_entry(stype, pop_val) for stype in pack]


def _pick_corner_richness(world_seed: int, cid: str, archetype: str,
                          rich_count: int, poor_count: int,
                          extra_poor: int = 0) -> str:
    # Global budget: max 3 poor cells, max 2 rich cells.
    # extra_poor = start cell (always poor, =1) + any neighbor poors already assigned.
    candidates = ["poor", "normal", "rich"]
    if rich_count >= 2:
        candidates = [c for c in candidates if c != "rich"]
    if poor_count + extra_poor >= 3:
        candidates = [c for c in candidates if c != "poor"]
    if not candidates:
        candidates = ["normal"]
    idx = choose_index(world_seed, f"rich:{cid}", len(candidates))
    richness = candidates[idx % len(candidates)]
    if richness == "rich":
        packs = SAVANNA_ARCHETYPES[archetype]["packs"]
        if not any(_is_contested(p) for p in packs):
            richness = "normal"
    return richness


def _pick_corner_pack(world_seed: int, cid: str, archetype: str, richness: str) -> list:
    packs = list(SAVANNA_ARCHETYPES[archetype]["packs"])
    if richness != "poor":
        filtered = [p for p in packs if len(p) > 0]
        if filtered:
            packs = filtered
    if richness == "rich":
        contested = [p for p in packs if _is_contested(p)]
        if contested:
            packs = contested
    idx = choose_index(world_seed, f"pack:{cid}", len(packs))
    return packs[idx % max(len(packs), 1)]


def _savanna_worldgen(world_seed: int) -> dict:
    """Return {cid: {archetype, richness, pack, resources, npc_species}} for all 9 cells."""
    START     = "c11"
    CORNERS   = ["c00", "c02", "c20", "c22"]
    NEIGHBORS = ["c01", "c10", "c12", "c21"]
    SLOTS     = ["safe_training", "leaf_offspec", "root_offspec", "wildcard"]

    result = {}

    # Start cell: fixed spec
    result[START] = {
        "archetype": "grassland",
        "richness":  "poor",
        "pack":      [],
        "resources": _savanna_resources("grassland", "poor"),
        "npc_species": [],
    }

    # Neighbors: slot permutation driven by seed
    slot_perm = _savanna_permute(world_seed, "slots", list(range(4)))
    for i, cid in enumerate(NEIGHBORS):
        slot_name = SLOTS[slot_perm[i]]
        opts = SAVANNA_SLOT_OPTIONS[slot_name]
        opt  = opts[choose_index(world_seed, f"slot_opt:{cid}", len(opts))]
        result[cid] = {
            "archetype":   opt["archetype"],
            "richness":    opt["richness"],
            "pack":        opt["pack"],
            "resources":   _savanna_resources(opt["archetype"], opt["richness"]),
            "npc_species": _build_npc_pack(opt["pack"], opt["richness"]),
        }

    # Corners: pick template, rotate, reflect; then assign richness+pack
    t_idx = choose_index(world_seed, "template", len(SAVANNA_TEMPLATES))
    rot   = choose_index(world_seed, "rot", 4)
    ref   = choose_index(world_seed, "ref", 2)

    raw_tmpl = SAVANNA_TEMPLATES[t_idx]
    transformed = {}
    for src, archetype in raw_tmpl.items():
        dst = _reflect_cid(src) if ref else src
        dst = _rotate_cid(dst, rot)
        if dst in CORNERS:
            transformed[dst] = archetype

    # count non-corner poors already assigned: start (c11, always poor) + any neighbor with richness=="poor"
    extra_poor = 1  # start cell
    for cid in NEIGHBORS:
        if result[cid]["richness"] == "poor":
            extra_poor += 1

    rich_count = 0
    poor_count = 0
    corner_data = {}
    for cid in CORNERS:
        archetype = transformed.get(cid, "grassland")
        richness  = _pick_corner_richness(world_seed, cid, archetype, rich_count, poor_count, extra_poor=extra_poor)
        if richness == "rich":  rich_count += 1
        elif richness == "poor": poor_count += 1
        pack = _pick_corner_pack(world_seed, cid, archetype, richness)
        corner_data[cid] = (archetype, richness, pack)

    # Enforce min_poor_cells=2 globally (start=poor counts as 1, so need 1+ poor corner)
    if poor_count == 0:
        for cid in CORNERS:
            archetype, richness, pack = corner_data[cid]
            if richness != "poor":
                richness = "poor"
                pack = _pick_corner_pack(world_seed, cid, archetype, "poor")
                corner_data[cid] = (archetype, richness, pack)
                break

    for cid, (archetype, richness, pack) in corner_data.items():
        result[cid] = {
            "archetype":   archetype,
            "richness":    richness,
            "pack":        pack,
            "resources":   _savanna_resources(archetype, richness),
            "npc_species": _build_npc_pack(pack, richness),
        }

    return result

# --- Legacy random worldgen (kept for reference, no longer called by init_new_run) ---

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

    start_cell = "c11"  # fixed start (savanna spec)

    # Savanna deterministic worldgen
    worldgen = _savanna_worldgen(world_seed)

    cells = {}
    for cid in all_cells:
        wg = worldgen[cid]
        cells[cid] = {
            "pos": {"x": int(cid[1]), "y": int(cid[2])},
            "archetype": wg["archetype"],
            "richness":  wg["richness"],
            "resources": wg["resources"],
            "npc_species": wg["npc_species"],
            "neighbors": [],
        }

    # 4-neighborhood links
    for cid in all_cells:
        x = int(cid[1]); y = int(cid[2])
        nbrs = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                nbrs.append(f"c{nx}{ny}")
        cells[cid]["neighbors"] = nbrs

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
        if amt > cap + 0.005:
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
    if happened and event_info and event_info.get("type") == "fluctuation":
        save_obj["state"].setdefault("active_effects", [])
        save_obj["state"]["active_effects"].append(event_info["eff"])
        events_out.append({"type": "fluctuation", "cell": event_info["cell"], "res": event_info["res"],
                           "mult": event_info["mult"], "ttl": event_info["ttl"]})

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
                if surv > float(RULES["extinction_eps"]):
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

    # remove extinct NPC entries from cells
    eps = float(RULES["extinction_eps"])
    for cell in save_obj["world"]["cells"].values():
        cell["npc_species"] = [n for n in cell.get("npc_species", []) if float(n.get("pop", 0.0)) > eps]

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
            rel = abs(d) / max(1.0, abs(float(h[-2])))
            if rel > 0.005:   # > 0.5% change → show arrow
                trend = "up" if d > 0 else "down"
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

    # active effects by cell for map overlay
    active_effect_cells = set()
    for eff in (s.get("state", {}).get("active_effects", []) or []):
        try:
            active_effect_cells.add(str(eff["cell"]))
        except Exception:
            pass
    active_effects_map = _apply_active_effects(s)  # {cid: {res: mult}}

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
            cells[cid]["archetype"] = cell.get("archetype")
            cells[cid]["richness"]  = cell.get("richness")
            cells[cid]["resources"] = {k: float(v) for k, v in cell.get("resources", {}).items()}
            cells[cid]["has_event"] = cid in active_effect_cells
            if cid in active_effect_cells:
                eff_res = _effective_resources(cell, active_effects_map.get(cid, {}))
                cells[cid]["resources_effective"] = {k: float(v) for k, v in eff_res.items()}
            npc_sum = []
            for npc in cell.get("npc_species", []):
                stype = npc.get("species_type")
                sp_data = SAVANNA_SPECIES.get(stype) if stype else None
                if sp_data and float(npc.get("pop", 0.0)) >= 0.5:
                    npc_sum.append({
                        "emoji":   sp_data["emoji"],
                        "ui_name": sp_data["ui_name"],
                        "pop":     float(npc["pop"]),
                    })
            cells[cid]["npc_summary"] = npc_sum

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
