import { SimConfig, WorldState, Region } from './types.js';
import { HexGrid } from './types.js';
import { chooseIndex, shuffleIndices } from './rng.js';
import { generateContinent, NEIGHBOR_DQ, NEIGHBOR_DR, coordKey, hexDistance } from './hexgrid.js';
import { BIOMES, getBiome } from './biomes.js';
import { SPECIES } from './species.js';

export function generateWorld(config: SimConfig): WorldState {
  const seed = config.worldSeed;
  const grid = generateContinent(config.totalHexes, seed, 11);
  generateRegions(grid, config.numRegions, config.minRegionSize, seed);
  mergeSmallRegions(grid, config.minRegionSize);
  fixDisconnectedRegions(grid);

  // Remap region IDs to 0..N-1
  const usedIds = new Set<number>();
  for (const li of grid.landIndices) {
    const rid = grid.tiles[li].regionId;
    if (rid >= 0) usedIds.add(rid);
  }
  const remap = new Map<number, number>();
  let next = 0;
  for (const old of [...usedIds].sort((a, b) => a - b)) remap.set(old, next++);
  for (const li of grid.landIndices) {
    const old = grid.tiles[li].regionId;
    if (old >= 0) grid.tiles[li].regionId = remap.get(old)!;
  }

  const numRegions = next;
  const regions: Region[] = Array.from({ length: numRegions }, (_, i) => makeEmptyRegion(i));

  for (const li of grid.landIndices) {
    const rid = grid.tiles[li].regionId;
    if (rid >= 0) {
      regions[rid].hexIndices.push(li);
      regions[rid].hexCount++;
    }
  }

  buildAdjacency(grid, regions);
  assignBiomes(regions, seed);
  computeBaseResources(regions, config.richnessPerHex);
  computeCentroids(grid, regions);
  seedSpecies(regions, config);

  const world: WorldState = {
    currentTick: 0,
    config,
    grid,
    regions,
    pendingMigrants: new Map(),
    migrationEdgesThisTick: [],
    metrics: {
      totalWorldBiomass: 0, aliveSpeciesCount: 0,
      fastestGrowing: '', fastestDeclining: '',
      speciesTotalPop: new Map(),
    },
    globalPopHistory: new Map(),
  };

  // Initial metrics snapshot
  updateMetrics(world);
  return world;
}

function makeEmptyRegion(id: number): Region {
  return {
    id, hexIndices: [], hexCount: 0, biomeId: '',
    neighbors: [], baseR1: 0, baseR2: 0, baseR3: 0,
    effR1: 0, effR2: 0, effR3: 0,
    populations: new Map(), tickBreakdown: new Map(),
    activeEffects: [], popHistory: new Map(),
    dominantSpecies: '', totalBiomass: 0, biomassTrend: 'flat',
    centroidX: 0, centroidY: 0,
  };
}

// ── GenerateRegions ──────────────────────────────────────────────────────────

function generateRegions(grid: HexGrid, numRegions: number, _minSize: number, seed: string): void {
  const L = grid.landIndices.length;
  if (L === 0) return;

  const minDist = Math.max(2, Math.floor(Math.sqrt(L / numRegions) * 0.5));
  const shuffled = shuffleIndices(seed, 'region_seeds', L);

  const seedTileIndices: number[] = [];
  for (const si of shuffled) {
    if (seedTileIndices.length >= numRegions) break;
    const tidx = grid.landIndices[si];
    const { q, r } = grid.tiles[tidx];
    let tooClose = false;
    for (const existing of seedTileIndices) {
      const { q: eq, r: er } = grid.tiles[existing];
      if (hexDistance(q, r, eq, er) < minDist) { tooClose = true; break; }
    }
    if (!tooClose) seedTileIndices.push(tidx);
  }
  // Fill remaining without distance constraint
  if (seedTileIndices.length < numRegions) {
    for (const si of shuffled) {
      if (seedTileIndices.length >= numRegions) break;
      const tidx = grid.landIndices[si];
      if (!seedTileIndices.includes(tidx)) seedTileIndices.push(tidx);
    }
  }

  const actual = seedTileIndices.length;
  for (let i = 0; i < actual; i++) grid.tiles[seedTileIndices[i]].regionId = i;

  const frontiers: number[][] = Array.from({ length: actual }, (_, i) => [seedTileIndices[i]]);

  let changed = true, round = 0;
  while (changed) {
    changed = false;
    for (let rId = 0; rId < actual; rId++) {
      if (frontiers[rId].length === 0) continue;

      const candidates = new Set<number>();
      for (const ft of frontiers[rId]) {
        const { q, r } = grid.tiles[ft];
        for (let e = 0; e < 6; e++) {
          const nk = coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]);
          const ni = grid.coordMap.get(nk);
          if (ni !== undefined && grid.tiles[ni].isLand && grid.tiles[ni].regionId < 0) {
            candidates.add(ni);
          }
        }
      }
      if (candidates.size === 0) { frontiers[rId] = []; continue; }

      // Pick most compact candidate
      let best = -1, bestScore = -1;
      const bestGroup: number[] = [];
      for (const ci of candidates) {
        const { q, r } = grid.tiles[ci];
        let score = 0;
        for (let e = 0; e < 6; e++) {
          const nk = coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]);
          const ni = grid.coordMap.get(nk);
          if (ni !== undefined && grid.tiles[ni].regionId === rId) score++;
        }
        if (score > bestScore) { bestScore = score; bestGroup.length = 0; }
        if (score === bestScore) bestGroup.push(ci);
      }
      const pick = chooseIndex(seed, `region_expand|${rId}|${round}`, bestGroup.length);
      best = bestGroup[pick];

      grid.tiles[best].regionId = rId;
      frontiers[rId].push(best);
      changed = true;
    }
    round++;
    if (round > L * 2) break;
  }

  // Assign any remaining unassigned land tiles to nearest seed
  for (const li of grid.landIndices) {
    if (grid.tiles[li].regionId >= 0) continue;
    const { q, r } = grid.tiles[li];
    let bestDist = Infinity, bestRegion = 0;
    for (let i = 0; i < actual; i++) {
      const { q: sq, r: sr } = grid.tiles[seedTileIndices[i]];
      const d = hexDistance(q, r, sq, sr);
      if (d < bestDist) { bestDist = d; bestRegion = i; }
    }
    grid.tiles[li].regionId = bestRegion;
  }
}

// ── MergeSmallRegions ────────────────────────────────────────────────────────

function mergeSmallRegions(grid: HexGrid, minSize: number): void {
  let merged = true;
  while (merged) {
    merged = false;
    const sizes = new Map<number, number>();
    for (const li of grid.landIndices) {
      const rid = grid.tiles[li].regionId;
      sizes.set(rid, (sizes.get(rid) ?? 0) + 1);
    }
    for (const [rid, sz] of sizes) {
      if (sz >= minSize) continue;
      const neighborSizes = new Map<number, number>();
      for (const li of grid.landIndices) {
        if (grid.tiles[li].regionId !== rid) continue;
        const { q, r } = grid.tiles[li];
        for (let e = 0; e < 6; e++) {
          const ni = grid.coordMap.get(coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]));
          if (ni !== undefined && grid.tiles[ni].isLand && grid.tiles[ni].regionId !== rid) {
            const nbRid = grid.tiles[ni].regionId;
            neighborSizes.set(nbRid, sizes.get(nbRid) ?? 0);
          }
        }
      }
      if (neighborSizes.size === 0) continue;
      let bestNb = -1, bestNbSz = -1;
      for (const [nbRid, nbSz] of neighborSizes) {
        if (nbSz > bestNbSz) { bestNbSz = nbSz; bestNb = nbRid; }
      }
      for (const li of grid.landIndices) {
        if (grid.tiles[li].regionId === rid) grid.tiles[li].regionId = bestNb;
      }
      merged = true;
      break;
    }
  }
}

// ── FixDisconnectedRegions ───────────────────────────────────────────────────

function fixDisconnectedRegions(grid: HexGrid): void {
  let anyFixed = true;
  while (anyFixed) {
    anyFixed = false;
    const regionTiles = new Map<number, number[]>();
    for (const li of grid.landIndices) {
      const rid = grid.tiles[li].regionId;
      const arr = regionTiles.get(rid);
      if (arr) arr.push(li); else regionTiles.set(rid, [li]);
    }
    for (const [rid, tileList] of regionTiles) {
      if (tileList.length <= 1) continue;
      const visited = new Set<number>();
      const queue = [tileList[0]];
      visited.add(tileList[0]);
      for (let qi = 0; qi < queue.length; qi++) {
        const { q, r } = grid.tiles[queue[qi]];
        for (let e = 0; e < 6; e++) {
          const ni = grid.coordMap.get(coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]));
          if (ni !== undefined && grid.tiles[ni].regionId === rid && !visited.has(ni)) {
            visited.add(ni); queue.push(ni);
          }
        }
      }
      for (const ti of tileList) {
        if (visited.has(ti)) continue;
        const { q, r } = grid.tiles[ti];
        for (let e = 0; e < 6; e++) {
          const ni = grid.coordMap.get(coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]));
          if (ni !== undefined && grid.tiles[ni].isLand && grid.tiles[ni].regionId !== rid) {
            grid.tiles[ti].regionId = grid.tiles[ni].regionId;
            anyFixed = true;
            break;
          }
        }
      }
    }
  }
}

// ── BuildAdjacency ───────────────────────────────────────────────────────────

function buildAdjacency(grid: HexGrid, regions: Region[]): void {
  const seen = new Set<number>();
  for (const li of grid.landIndices) {
    const rId = grid.tiles[li].regionId;
    if (rId < 0) continue;
    const { q, r } = grid.tiles[li];
    for (let e = 0; e < 6; e++) {
      const ni = grid.coordMap.get(coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]));
      if (ni === undefined || !grid.tiles[ni].isLand) continue;
      const nbRid = grid.tiles[ni].regionId;
      if (nbRid < 0 || nbRid === rId) continue;
      const lo = Math.min(rId, nbRid), hi = Math.max(rId, nbRid);
      const pairKey = lo * 10000 + hi;
      if (seen.has(pairKey)) continue;
      seen.add(pairKey);
      if (!regions[rId].neighbors.includes(nbRid)) regions[rId].neighbors.push(nbRid);
      if (!regions[nbRid].neighbors.includes(rId)) regions[nbRid].neighbors.push(rId);
    }
  }
}

// ── AssignBiomes ─────────────────────────────────────────────────────────────

function assignBiomes(regions: Region[], seed: string): void {
  const n = BIOMES.length;
  for (let i = 0; i < regions.length; i++) {
    regions[i].biomeId = BIOMES[chooseIndex(seed, `biome_assign|${i}`, n)].id;
  }
}

// ── ComputeBaseResources ─────────────────────────────────────────────────────

function computeBaseResources(regions: Region[], richnessPerHex: number): void {
  for (const region of regions) {
    const biome = getBiome(region.biomeId);
    const total = region.hexCount * richnessPerHex;
    const sum = biome.ratioR1 + biome.ratioR2 + biome.ratioR3 || 1;
    region.baseR1 = total * biome.ratioR1 / sum;
    region.baseR2 = total * biome.ratioR2 / sum;
    region.baseR3 = total * biome.ratioR3 / sum;
    region.effR1 = region.baseR1;
    region.effR2 = region.baseR2;
    region.effR3 = region.baseR3;
  }
}

// ── ComputeCentroids ─────────────────────────────────────────────────────────

function computeCentroids(grid: HexGrid, regions: Region[]): void {
  for (const region of regions) {
    let sx = 0, sy = 0;
    for (const hi of region.hexIndices) {
      sx += grid.tiles[hi].wx;
      sy += grid.tiles[hi].wy;
    }
    region.centroidX = sx / region.hexCount;
    region.centroidY = sy / region.hexCount;
  }
}

// ── SeedSpecies ──────────────────────────────────────────────────────────────

function seedSpecies(regions: Region[], config: SimConfig): void {
  const seed = config.worldSeed;
  for (let i = 0; i < regions.length; i++) {
    const region = regions[i];
    const biome = getBiome(region.biomeId);
    const nonEmpty = biome.packs.filter(p => p.length > 0);
    if (nonEmpty.length === 0) continue;
    const pack = nonEmpty[chooseIndex(seed, `species_pack|${i}`, nonEmpty.length)];
    const startPop = region.hexCount * config.startPopDensity;
    for (const spId of pack) region.populations.set(spId, startPop);
  }
}

// ── UpdateMetrics (used by simulation.ts too) ────────────────────────────────

export function updateMetrics(world: WorldState): void {
  const { regions, config } = world;
  const m = world.metrics;
  m.speciesTotalPop = new Map();
  m.totalWorldBiomass = 0;

  for (const region of regions) {
    region.totalBiomass = 0;
    for (const [sp, pop] of region.populations) {
      region.totalBiomass += pop;
      m.speciesTotalPop.set(sp, (m.speciesTotalPop.get(sp) ?? 0) + pop);
      m.totalWorldBiomass += pop;
    }
    region.dominantSpecies = findDominant(region);

    // Per-species pop history
    for (const [sp, pop] of region.populations) {
      const hist = region.popHistory.get(sp) ?? [];
      hist.push(pop);
      if (hist.length > config.trendHistoryLen) hist.shift();
      region.popHistory.set(sp, hist);
      const bd = region.tickBreakdown.get(sp);
      if (bd && hist.length > config.trendWindowShort) {
        const old = hist[hist.length - 1 - config.trendWindowShort];
        if (old > 0) {
          const delta = (pop - old) / old;
          bd.trend = delta > config.trendThreshold ? 'up' : delta < -config.trendThreshold ? 'down' : 'flat';
        }
      }
    }
    // Clean extinct from history
    for (const sp of region.popHistory.keys()) {
      if (!region.populations.has(sp)) region.popHistory.delete(sp);
    }

    // Region biomass trend
    let curTotal = region.totalBiomass, pastTotal = 0, hasPast = false;
    for (const hist of region.popHistory.values()) {
      if (hist.length > config.trendWindowShort) {
        pastTotal += hist[hist.length - 1 - config.trendWindowShort];
        hasPast = true;
      }
    }
    if (hasPast && pastTotal > 0) {
      const delta = (curTotal - pastTotal) / pastTotal;
      region.biomassTrend = delta > config.trendThreshold ? 'up' : delta < -config.trendThreshold ? 'down' : 'flat';
    }
  }

  m.aliveSpeciesCount = m.speciesTotalPop.size;

  // Global pop history
  for (const [sp, pop] of m.speciesTotalPop) {
    const hist = world.globalPopHistory.get(sp) ?? [];
    hist.push(pop);
    if (hist.length > config.trendHistoryLen) hist.shift();
    world.globalPopHistory.set(sp, hist);
  }
  // Clean extinct
  for (const sp of world.globalPopHistory.keys()) {
    if (!m.speciesTotalPop.has(sp)) world.globalPopHistory.delete(sp);
  }

  // Fastest growing / declining
  let bestGrowth = -Infinity, worstDecline = Infinity;
  m.fastestGrowing = ''; m.fastestDeclining = '';
  for (const [sp, hist] of world.globalPopHistory) {
    if (hist.length > config.trendWindowShort) {
      const old = hist[hist.length - 1 - config.trendWindowShort];
      const cur = hist[hist.length - 1];
      if (old > 0) {
        const delta = (cur - old) / old;
        if (delta > bestGrowth) { bestGrowth = delta; m.fastestGrowing = sp; }
        if (delta < worstDecline) { worstDecline = delta; m.fastestDeclining = sp; }
      }
    }
  }
}

function findDominant(region: Region): string {
  let best = '', bestPop = 0;
  for (const [sp, pop] of region.populations) {
    if (pop > bestPop) { bestPop = pop; best = sp; }
  }
  return best;
}

export { SPECIES };
