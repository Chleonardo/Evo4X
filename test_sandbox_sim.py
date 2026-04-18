"""
Standalone sandbox simulation test.
Runs the new spatial sim logic (births, allocation, starvation, migration)
on 20 regions for 300+ ticks to validate balance and find imbalanced species.

No UE5 dependency — pure Python, same math as the C++ SimulationCore.
"""

import hashlib
import math
import json
from collections import defaultdict

# ============================================================================
# RNG — identical to engine
# ============================================================================

def rand01(seed, key):
    h = hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()
    return int(h[:8], 16) / (2**32)

def choose_index(seed, key, n):
    if n <= 0:
        return 0
    return min(n - 1, int(rand01(seed, key) * n))

def shuffle_indices(seed, key, n):
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        j = choose_index(seed, f"{key}|shuffle|{i}", i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return indices

# ============================================================================
# Data
# ============================================================================

with open("data/species.json", encoding="utf-8") as f:
    SPECIES_RAW = json.load(f)

with open("data/archetypes.json", encoding="utf-8") as f:
    ARCHETYPES = json.load(f)

# Add radiation values
RADIATION = {
    "gazelle": 8, "impala": 6, "zebra": 5, "buffalo": 3,
    "giraffe": 4, "elephant": 2, "warthog": 6, "mole_rat": 5
}

SPECIES = {}
for sid, sdata in SPECIES_RAW.items():
    SPECIES[sid] = {
        "r": sdata["stats"]["r"],
        "b": [sdata["stats"]["b"]["R1"], sdata["stats"]["b"]["R2"], sdata["stats"]["b"]["R3"]],
        "radiation": RADIATION[sid],
        "name": sdata["ui_name"]
    }

# Balance overrides for sandbox v1
SPECIES["warthog"]["r"] = 1.20
SPECIES["zebra"]["b"] = [1.1, 0.0, 0.0]

BIOME_IDS = list(ARCHETYPES.keys())

# ============================================================================
# World Generation
# ============================================================================

def generate_world(seed, num_regions=20, richness_per_hex=10, start_pop_density=2):
    regions = []

    for i in range(num_regions):
        # Assign biome
        biome_id = BIOME_IDS[choose_index(seed, f"biome_assign|{i}", len(BIOME_IDS))]
        biome = ARCHETYPES[biome_id]

        # Region size: 15-35 hexes (simulating ~500/20 with variance)
        hex_count = 20 + choose_index(seed, f"region_size|{i}", 16) - 8  # 12..27
        hex_count = max(5, hex_count)

        # Resources
        ratio_sum = biome["ratio"]["R1"] + biome["ratio"]["R2"] + biome["ratio"]["R3"]
        total_richness = hex_count * richness_per_hex
        r1 = total_richness * biome["ratio"]["R1"] / ratio_sum if ratio_sum > 0 else 0
        r2 = total_richness * biome["ratio"]["R2"] / ratio_sum if ratio_sum > 0 else 0
        r3 = total_richness * biome["ratio"]["R3"] / ratio_sum if ratio_sum > 0 else 0

        # Pick species pack
        non_empty_packs = [p for p in biome["packs"] if len(p) > 0]
        populations = {}
        if non_empty_packs:
            pack_idx = choose_index(seed, f"species_pack|{i}", len(non_empty_packs))
            pack = non_empty_packs[pack_idx]
            start_pop = hex_count * start_pop_density
            for sp in pack:
                populations[sp] = start_pop

        # Neighbors: build later
        regions.append({
            "id": i,
            "biome": biome_id,
            "hex_count": hex_count,
            "resources": [r1, r2, r3],
            "populations": populations,
            "neighbors": [],
        })

    # Build adjacency: ring topology + some cross-links for realism
    for i in range(num_regions):
        # Ring neighbors
        regions[i]["neighbors"].append((i - 1) % num_regions)
        regions[i]["neighbors"].append((i + 1) % num_regions)
        # Add 1-2 extra neighbors for connectivity
        extra = choose_index(seed, f"extra_nb|{i}", num_regions)
        if extra != i and extra not in regions[i]["neighbors"]:
            regions[i]["neighbors"].append(extra)
            if i not in regions[extra]["neighbors"]:
                regions[extra]["neighbors"].append(i)

    return regions

# ============================================================================
# Simulation Tick
# ============================================================================

def simulate_tick(regions, tick, seed, extinction_eps=1.0):
    pending_migrants = defaultdict(lambda: defaultdict(float))  # region_id -> species -> count
    migration_log = []

    # ── Phase 1: per-region ──
    for region in regions:
        pops = region["populations"]
        if not pops:
            continue

        res = region["resources"]

        # Step 2: Births
        n_pre = {}
        births = {}
        for sp, pop in pops.items():
            b = pop * SPECIES[sp]["r"]
            births[sp] = b
            n_pre[sp] = pop + b

        # Step 3: Allocation
        food_allocated = {sp: 0.0 for sp in n_pre}
        for r_idx in range(3):
            avail = res[r_idx]
            if avail <= 0:
                continue

            weights = {}
            total_w = 0.0
            for sp, npre in n_pre.items():
                w = npre * SPECIES[sp]["b"][r_idx]
                if w > 0:
                    weights[sp] = w
                    total_w += w

            if total_w <= 0:
                continue

            for sp, w in weights.items():
                food_allocated[sp] += avail * w / total_w

        # Steps 4-5: Starving newborns + migration
        outgoing = {}
        for sp in list(n_pre.keys()):
            starving_total = max(0, n_pre[sp] - food_allocated[sp])
            starving_newborns = min(births[sp], starving_total)
            starving_pairs = int(starving_newborns / 2)

            if starving_pairs <= 0 or len(region["neighbors"]) == 0:
                outgoing[sp] = 0
                continue

            # Bernoulli loop
            radiation_chance = SPECIES[sp]["radiation"] / 100.0
            num_expeditions = 0
            for pair_idx in range(starving_pairs):
                key = f"migrate|{tick}|{region['id']}|{sp}|{pair_idx}"
                if rand01(seed, key) < radiation_chance:
                    num_expeditions += 1

            total_migrants = num_expeditions * 2
            outgoing[sp] = total_migrants

            if num_expeditions > 0:
                # Distribute among neighbors
                N = len(region["neighbors"])
                base = num_expeditions // N
                rem = num_expeditions % N

                for nb_id in region["neighbors"]:
                    if base > 0:
                        pending_migrants[nb_id][sp] += base * 2

                if rem > 0:
                    shuffled = shuffle_indices(seed, f"mig_rem|{tick}|{region['id']}|{sp}", N)
                    for k in range(rem):
                        nb_id = region["neighbors"][shuffled[k]]
                        pending_migrants[nb_id][sp] += 2

                migration_log.append((region["id"], sp, total_migrants))

        # Step 6: Commit starvation
        new_pops = {}
        for sp in n_pre:
            effective_npre = n_pre[sp] - outgoing.get(sp, 0)
            survivors = min(effective_npre, food_allocated[sp])
            survivors = max(0, survivors)
            new_pops[sp] = survivors

        region["populations"] = new_pops

    # ── Phase 2: Apply incoming migrants ──
    for region_id, species_migrants in pending_migrants.items():
        for sp, count in species_migrants.items():
            if sp not in regions[region_id]["populations"]:
                regions[region_id]["populations"][sp] = 0
            regions[region_id]["populations"][sp] += count

    # ── Phase 3: Extinction cleanup ──
    for region in regions:
        to_remove = [sp for sp, pop in region["populations"].items() if pop <= extinction_eps]
        for sp in to_remove:
            del region["populations"][sp]

    return migration_log

# ============================================================================
# Run simulation
# ============================================================================

def run_simulation(seed="42", ticks=300, num_regions=20):
    print(f"=== Sandbox Simulation Test ===")
    print(f"Seed: {seed}, Ticks: {ticks}, Regions: {num_regions}")
    print()

    regions = generate_world(seed, num_regions)

    # Print initial state
    print("── Initial State ──")
    global_pops = defaultdict(float)
    for r in regions:
        for sp, pop in r["populations"].items():
            global_pops[sp] += pop

    for sp in sorted(global_pops.keys(), key=lambda s: -global_pops[s]):
        print(f"  {SPECIES[sp]['name']:12s}: {global_pops[sp]:8.0f}  (r={SPECIES[sp]['r']:.2f}, rad={SPECIES[sp]['radiation']}%)")
    print(f"  Total biomass: {sum(global_pops.values()):.0f}")
    print()

    # Track history
    history = defaultdict(list)
    total_history = []

    # Simulate
    milestones = [10, 25, 50, 100, 150, 200, 300]

    for tick in range(ticks):
        migration_log = simulate_tick(regions, tick, seed)

        # Collect global stats
        global_pops = defaultdict(float)
        for r in regions:
            for sp, pop in r["populations"].items():
                global_pops[sp] += pop

        for sp in SPECIES:
            history[sp].append(global_pops.get(sp, 0))
        total_history.append(sum(global_pops.values()))

        if (tick + 1) in milestones:
            print(f"── Tick {tick + 1} ──")
            alive = {sp: pop for sp, pop in global_pops.items() if pop > 0}
            for sp in sorted(alive.keys(), key=lambda s: -alive[s]):
                # Count regions occupied
                occupied = sum(1 for r in regions if sp in r["populations"] and r["populations"][sp] > 0)
                print(f"  {SPECIES[sp]['name']:12s}: {alive[sp]:10.1f}  ({occupied}/{num_regions} regions)")

            extinct = [sp for sp in SPECIES if global_pops.get(sp, 0) == 0 and any(sp in h for h in [history])]
            total = sum(alive.values())
            print(f"  Total biomass: {total:.0f}")
            print(f"  Alive species: {len(alive)}")

            # Migration activity
            total_migrants = sum(m[2] for m in migration_log)
            print(f"  Migrations this tick: {total_migrants:.0f}")
            print()

    # Final analysis
    print("=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)

    final_pops = defaultdict(float)
    species_regions = defaultdict(int)
    for r in regions:
        for sp, pop in r["populations"].items():
            final_pops[sp] += pop
            species_regions[sp] += 1

    print(f"\nSurviving species: {len(final_pops)} / {len(SPECIES)}")
    print(f"Total biomass: {sum(final_pops.values()):.0f}")
    print()

    print("Species ranking:")
    for sp in sorted(final_pops.keys(), key=lambda s: -final_pops[s]):
        share = final_pops[sp] / sum(final_pops.values()) * 100 if sum(final_pops.values()) > 0 else 0
        start = history[sp][0] if history[sp] else 0
        growth = (final_pops[sp] / start - 1) * 100 if start > 0 else float('inf')
        print(f"  {SPECIES[sp]['name']:12s}: {final_pops[sp]:10.1f} ({share:5.1f}%)  "
              f"in {species_regions[sp]:2d}/{num_regions} regions  "
              f"growth: {growth:+.0f}%")

    extinct = [sp for sp in SPECIES if sp not in final_pops]
    if extinct:
        print(f"\nExtinct species: {', '.join(SPECIES[sp]['name'] for sp in extinct)}")

    # Dominance check
    print("\n── Balance Warnings ──")
    total = sum(final_pops.values())
    for sp, pop in final_pops.items():
        share = pop / total * 100 if total > 0 else 0
        if share > 40:
            print(f"  WARNING: {SPECIES[sp]['name']} dominates with {share:.1f}% world share!")
        if species_regions[sp] >= num_regions * 0.8:
            print(f"  WARNING: {SPECIES[sp]['name']} spread to {species_regions[sp]}/{num_regions} regions (pandemic?)")

    if not any(pop / total * 100 > 40 for pop in final_pops.values()):
        print("  No single species dominates (>40%). Balance looks reasonable.")

    return history, final_pops

if __name__ == "__main__":
    run_simulation("42", 300, 20)
