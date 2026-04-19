import { WorldState, Region, SpeciesInRegion, MigrationEdge } from './types.js';
import { rand01, shuffleIndices } from './rng.js';
import { getSpecies } from './species.js';
import { updateMetrics } from './worldgen.js';

export function simulateTick(world: WorldState): void {
  const seed = world.config.worldSeed;
  const tick = world.currentTick;

  world.pendingMigrants = new Map();
  world.migrationEdgesThisTick = [];

  // ── Phase 1: per-region ──────────────────────────────────────────────────
  for (const region of world.regions) {
    region.tickBreakdown = new Map();

    applyActiveEffects(region);
    computeBirths(region);
    allocateResources(region);
    computeMigration(world, region, seed, tick);
    commitStarvation(region, world.config.extinctionEps);
  }

  // ── Phase 2: global ──────────────────────────────────────────────────────
  applyIncomingMigrants(world);
  extinctionCleanup(world);

  for (const region of world.regions) tickDownEffects(region);

  updateMetrics(world);
  world.currentTick++;
}

// ── Step 1 ───────────────────────────────────────────────────────────────────

function applyActiveEffects(region: Region): void {
  region.effR1 = region.baseR1;
  region.effR2 = region.baseR2;
  region.effR3 = region.baseR3;
  for (const eff of region.activeEffects) {
    region.effR1 *= eff.multR1;
    region.effR2 *= eff.multR2;
    region.effR3 *= eff.multR3;
  }
}

// ── Step 2 ───────────────────────────────────────────────────────────────────

function computeBirths(region: Region): void {
  for (const [sp, pop] of region.populations) {
    const spec = getSpecies(sp);
    const bd: SpeciesInRegion = {
      population: pop, popPreBirth: pop * (1 + spec.r),
      foodAllocated: 0, starvingTotal: 0, starvingNewborns: 0,
      outgoingMigrants: 0, incomingMigrants: 0, trend: 'flat',
    };
    region.tickBreakdown.set(sp, bd);
  }
}

// ── Step 3 ───────────────────────────────────────────────────────────────────

function allocateResources(region: Region): void {
  const avail = [region.effR1, region.effR2, region.effR3];
  const bWeights = [
    (sp: string) => getSpecies(sp).bR1,
    (sp: string) => getSpecies(sp).bR2,
    (sp: string) => getSpecies(sp).bR3,
  ];

  for (let ri = 0; ri < 3; ri++) {
    if (avail[ri] <= 0) continue;
    let totalW = 0;
    const weights: [string, number][] = [];
    for (const [sp, bd] of region.tickBreakdown) {
      const w = bd.popPreBirth * bWeights[ri](sp);
      if (w > 0) { weights.push([sp, w]); totalW += w; }
    }
    if (totalW <= 0) continue;
    for (const [sp, w] of weights) {
      region.tickBreakdown.get(sp)!.foodAllocated += avail[ri] * w / totalW;
    }
  }
}

// ── Step 4 ───────────────────────────────────────────────────────────────────

function computeMigration(world: WorldState, region: Region, seed: string, tick: number): void {
  const rid = region.id;
  const nbCount = region.neighbors.length;

  for (const [sp, bd] of region.tickBreakdown) {
    const spec = getSpecies(sp);
    const births = bd.popPreBirth - bd.population;
    bd.starvingTotal = Math.max(0, bd.popPreBirth - bd.foodAllocated);
    bd.starvingNewborns = Math.min(births, bd.starvingTotal);

    const starvingPairs = Math.floor(bd.starvingNewborns / 2);
    if (starvingPairs <= 0 || nbCount === 0) continue;

    const radiationChance = spec.radiation / 100;
    let numExpeditions = 0;
    for (let p = 0; p < starvingPairs; p++) {
      if (rand01(seed, `migrate|${tick}|${rid}|${sp}|${p}`) < radiationChance) numExpeditions++;
    }
    if (numExpeditions === 0) continue;

    bd.outgoingMigrants = numExpeditions * 2;

    const base = Math.floor(numExpeditions / nbCount);
    const rem  = numExpeditions % nbCount;

    for (let i = 0; i < nbCount; i++) {
      const count = base * 2;
      if (count <= 0) continue;
      addMigrants(world, region.neighbors[i], sp, count);
      addEdge(world.migrationEdgesThisTick, rid, region.neighbors[i], sp, count);
    }

    if (rem > 0) {
      const shuffled = shuffleIndices(seed, `mig_rem|${tick}|${rid}|${sp}`, nbCount);
      for (let i = 0; i < rem; i++) {
        const nbRid = region.neighbors[shuffled[i]];
        addMigrants(world, nbRid, sp, 2);
        addEdge(world.migrationEdgesThisTick, rid, nbRid, sp, 2);
      }
    }
  }
}

function addMigrants(world: WorldState, toRegion: number, sp: string, count: number): void {
  let m = world.pendingMigrants.get(toRegion);
  if (!m) { m = new Map(); world.pendingMigrants.set(toRegion, m); }
  m.set(sp, (m.get(sp) ?? 0) + count);
}

function addEdge(edges: MigrationEdge[], from: number, to: number, sp: string, count: number): void {
  const existing = edges.find(e => e.fromRegion === from && e.toRegion === to && e.speciesId === sp);
  if (existing) { existing.count += count; } else {
    edges.push({ fromRegion: from, toRegion: to, speciesId: sp, count });
  }
}

// ── Step 5 ───────────────────────────────────────────────────────────────────

function commitStarvation(region: Region, _eps: number): void {
  for (const [sp, bd] of region.tickBreakdown) {
    const effNPre = bd.popPreBirth - bd.outgoingMigrants;
    const survivors = Math.max(0, Math.min(effNPre, bd.foodAllocated));
    region.populations.set(sp, survivors);
  }
}

// ── Step 6 ───────────────────────────────────────────────────────────────────

function applyIncomingMigrants(world: WorldState): void {
  for (const [rid, migrants] of world.pendingMigrants) {
    if (rid < 0 || rid >= world.regions.length) continue;
    const region = world.regions[rid];
    for (const [sp, count] of migrants) {
      region.populations.set(sp, (region.populations.get(sp) ?? 0) + count);
      const bd = region.tickBreakdown.get(sp);
      if (bd) bd.incomingMigrants += count;
    }
  }
}

// ── Step 7 ───────────────────────────────────────────────────────────────────

function extinctionCleanup(world: WorldState): void {
  const eps = world.config.extinctionEps;
  for (const region of world.regions) {
    for (const [sp, pop] of region.populations) {
      if (pop <= eps) {
        region.populations.delete(sp);
        region.tickBreakdown.delete(sp);
      }
    }
  }
}

// ── Step 8 ───────────────────────────────────────────────────────────────────

function tickDownEffects(region: Region): void {
  for (let i = region.activeEffects.length - 1; i >= 0; i--) {
    region.activeEffects[i].ttl--;
    if (region.activeEffects[i].ttl <= 0) region.activeEffects.splice(i, 1);
  }
}
