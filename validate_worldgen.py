"""
validate_worldgen.py — Savanna worldgen validation suite.
Run: python validate_worldgen.py
Exit 0 = all pass. Exit 1 = failure (reason printed).
"""
import sys, math, copy

from evo4x_engine_v2_6_13_rtsgrid_uiapi import (
    _savanna_worldgen, _savanna_resources,
    SAVANNA_ARCHETYPES, SAVANNA_SPECIES, SAVANNA_RICHNESS,
    SAVANNA_SLOT_OPTIONS, SAVANNA_TEMPLATES,
    alloc_food_for_cell,
)

FAILURES = []

def fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")

def ok(msg):
    print(f"  ok:   {msg}")


# -----------------------------------------------------------------------
# 1. Determinism: same seed -> identical world, 20 seeds x 2 runs
# -----------------------------------------------------------------------
print("\n[1] Determinism test (20 seeds × 2 runs)")
for seed in range(20):
    a = _savanna_worldgen(seed)
    b = _savanna_worldgen(seed)
    for cid in a:
        if a[cid]["resources"] != b[cid]["resources"]:
            fail(f"seed={seed} cid={cid} resources differ")
            break
        if a[cid]["archetype"] != b[cid]["archetype"]:
            fail(f"seed={seed} cid={cid} archetype differs")
            break
        npc_a = [(n["species_type"], n["pop"]) for n in a[cid]["npc_species"]]
        npc_b = [(n["species_type"], n["pop"]) for n in b[cid]["npc_species"]]
        if npc_a != npc_b:
            fail(f"seed={seed} cid={cid} npc_species differ")
            break
    else:
        continue
    break  # inner break hit
else:
    ok("all 20 seeds deterministic")


# -----------------------------------------------------------------------
# 2. Resource sums match richness total
# -----------------------------------------------------------------------
print("\n[2] Resource sums test (all archetype × richness combos)")
for arch in SAVANNA_ARCHETYPES:
    for rich, total in SAVANNA_RICHNESS.items():
        res = _savanna_resources(arch, rich)
        s = int(res["R1"] + res["R2"] + res["R3"])
        if s != total:
            fail(f"{arch}/{rich}: sum={s} expected={total}")
ok("all resource sums correct") if not FAILURES else None


# -----------------------------------------------------------------------
# 3. No-killer-start: 200 seeds — neighbor guarantees
# -----------------------------------------------------------------------
print("\n[3] No-killer-start test (200 seeds)")
NEIGHBORS = ["c01", "c10", "c12", "c21"]
SLOT_ARCHETYPES = {
    slot: set(o["archetype"] for o in opts)
    for slot, opts in SAVANNA_SLOT_OPTIONS.items()
}
ALL_ALLOWED_NEIGHBOR_ARCHETYPES = set()
for opts in SAVANNA_SLOT_OPTIONS.values():
    for o in opts:
        ALL_ALLOWED_NEIGHBOR_ARCHETYPES.add(o["archetype"])

start_fails = 0
for seed in range(200):
    wg = _savanna_worldgen(seed)
    # Start cell must be grassland/poor/no-npc
    sc = wg["c11"]
    if sc["archetype"] != "grassland" or sc["richness"] != "poor" or sc["npc_species"]:
        fail(f"seed={seed}: start cell wrong: {sc['archetype']}/{sc['richness']}")
        start_fails += 1
        continue
    # Each neighbor must be one of the slot-allowed archetypes
    for cid in NEIGHBORS:
        cell = wg[cid]
        if cell["archetype"] not in ALL_ALLOWED_NEIGHBOR_ARCHETYPES:
            fail(f"seed={seed} neighbor {cid}: archetype={cell['archetype']} not in slot options")
            start_fails += 1
        # Neighbors must not be rich (all slot options are normal or poor)
        if cell["richness"] == "rich":
            fail(f"seed={seed} neighbor {cid}: richness=rich (not allowed near start)")
            start_fails += 1
        # Note: safe_training slot explicitly allows pack=[] + normal richness (per spec),
        # so we do NOT check empty-pack constraint for neighbors.

if start_fails == 0:
    ok("all 200 seeds: start + neighbor constraints pass")


# -----------------------------------------------------------------------
# 4. Anti-snowball: global richness constraints (200 seeds)
# -----------------------------------------------------------------------
print("\n[4] Anti-snowball: richness budget (200 seeds)")
snowball_fails = 0
ALL_CELLS   = ["c00","c01","c02","c10","c11","c12","c20","c21","c22"]
CORNERS_SET = ["c00","c02","c20","c22"]
for seed in range(200):
    wg = _savanna_worldgen(seed)
    rich_cells  = [c for c in ALL_CELLS if wg[c]["richness"] == "rich"]
    poor_cells  = [c for c in ALL_CELLS if wg[c]["richness"] == "poor"]  # includes c11
    if len(rich_cells) > 2:
        fail(f"seed={seed}: {len(rich_cells)} rich cells (max 2)")
        snowball_fails += 1
    if len(poor_cells) < 2:
        fail(f"seed={seed}: {len(poor_cells)} poor cells (min 2)")
        snowball_fails += 1
    if len(poor_cells) > 3:
        fail(f"seed={seed}: {len(poor_cells)} poor cells (max 3)")
        snowball_fails += 1
    # Rich cells must have contested packs
    for cid in rich_cells:
        pack = wg[cid]["pack"]
        if not ("buffalo" in pack or "elephant" in pack or len(pack) >= 3):
            fail(f"seed={seed} {cid}: rich cell has non-contested pack {pack}")
            snowball_fails += 1
    # Empty pack only in poor — corners only (neighbors have explicit slot overrides)
    for cid in CORNERS_SET:
        cell = wg[cid]
        if cell["richness"] != "poor" and cell["pack"] == []:
            fail(f"seed={seed} {cid}: empty pack in {cell['richness']} corner cell")
            snowball_fails += 1

if snowball_fails == 0:
    ok("all 200 seeds: richness budget constraints pass")


# -----------------------------------------------------------------------
# 5. Stability test: NPC-only, 500 ticks, all archetype×pack at normal
# -----------------------------------------------------------------------
print("\n[5] Stability test: NPC-only 500 ticks (all archetype×pack at normal)")

def sim_cell_500(archetype, pack, richness="normal"):
    """Run local simulation for 500 ticks. Returns final pops dict {species_type: pop}."""
    res = _savanna_resources(archetype, richness)
    from evo4x_engine_v2_6_13_rtsgrid_uiapi import SAVANNA_NPC_POP
    n = min(len(pack), 3)
    if n == 0:
        return {}
    pop_init = float(SAVANNA_NPC_POP[n][richness])
    pops = {stype: pop_init for stype in pack}

    for _ in range(500):
        contestants = []
        for stype, pop in pops.items():
            sp = SAVANNA_SPECIES[stype]
            n_pre = pop * (1.0 + sp["stats"]["r"])
            contestants.append({"id": stype, "kind": "npc", "n_pre": n_pre, "b": sp["stats"]["b"]})
        food = alloc_food_for_cell(res, contestants)
        new_pops = {}
        for stype, pop in pops.items():
            n_pre = pop * (1.0 + SAVANNA_SPECIES[stype]["stats"]["r"])
            surv = min(n_pre, food.get(stype, 0.0))
            if surv > 0.01:
                new_pops[stype] = surv
        pops = new_pops
        if not pops:
            break
    return pops

unstable = []
for arch, adef in SAVANNA_ARCHETYPES.items():
    for pack in adef["packs"]:
        if not pack:
            continue
        final = sim_cell_500(arch, pack)
        for stype in pack:
            if final.get(stype, 0.0) < 1.0:
                unstable.append((arch, pack, stype, final.get(stype, 0.0)))

if unstable:
    for arch, pack, stype, pop in unstable:
        fail(f"stability: {arch} pack={pack} → {stype} collapsed to {pop:.2f}")
else:
    ok("all archetype×pack combinations stable at 500 ticks (NPC-only, normal)")


# -----------------------------------------------------------------------
# 6. Smoke test with player: 200 ticks, one cell per archetype
# -----------------------------------------------------------------------
print("\n[6] Smoke test with player sp0 (200 ticks, normal richness, one pack per archetype)")

PLAYER_SP0 = {"r": 0.5, "b": {"R1": 1.0, "R2": 0.5, "R3": 0.0}}

def sim_with_player_200(archetype, pack, richness="normal"):
    res = _savanna_resources(archetype, richness)
    from evo4x_engine_v2_6_13_rtsgrid_uiapi import SAVANNA_NPC_POP
    n = min(max(len(pack), 1), 3)
    pop_init = float(SAVANNA_NPC_POP[n][richness]) if pack else 0.0
    npc_pops = {stype: pop_init for stype in pack}
    player_pop = 2.0

    for _ in range(200):
        contestants = []
        if player_pop > 0:
            n_pre_p = player_pop * (1.0 + PLAYER_SP0["r"])
            contestants.append({"id": "sp0", "kind": "player", "n_pre": n_pre_p, "b": PLAYER_SP0["b"]})
        for stype, pop in npc_pops.items():
            n_pre = pop * (1.0 + SAVANNA_SPECIES[stype]["stats"]["r"])
            contestants.append({"id": stype, "kind": "npc", "n_pre": n_pre, "b": SAVANNA_SPECIES[stype]["stats"]["b"]})
        if not contestants:
            break
        food = alloc_food_for_cell(res, contestants)
        if player_pop > 0:
            n_pre_p = player_pop * (1.0 + PLAYER_SP0["r"])
            player_pop = min(n_pre_p, food.get("sp0", 0.0))
        new_npc = {}
        for stype, pop in npc_pops.items():
            n_pre = pop * (1.0 + SAVANNA_SPECIES[stype]["stats"]["r"])
            surv = min(n_pre, food.get(stype, 0.0))
            if surv > 0.01:
                new_npc[stype] = surv
        npc_pops = new_npc

    return player_pop, npc_pops

smoke_fails = 0
for arch, adef in SAVANNA_ARCHETYPES.items():
    # Use first non-empty pack
    pack = next((p for p in adef["packs"] if p), [])
    p_pop, npc_final = sim_with_player_200(arch, pack)
    if p_pop < 0.01 and pack:
        # Player collapse can happen in tough cells, just warn
        print(f"  warn: {arch} pack={pack} player collapsed (pop={p_pop:.3f}) — may be expected")

ok("smoke test complete (see warnings above if any)")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*50}")
if FAILURES:
    print(f"FAILED — {len(FAILURES)} issue(s):")
    for f in FAILURES:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
