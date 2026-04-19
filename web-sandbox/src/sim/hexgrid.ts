import { HexGrid, HexTile } from './types.js';
import { rand01, chooseIndex } from './rng.js';

// Flat-top hex.
// World coords: wx = size * 1.5 * q,  wy = size * sqrt(3) * (r + q*0.5)
// Corner i (flat-top): angle = 60° * i  →  (cos, sin)
// Neighbor directions per edge index 0..5 (edge i = border shared with neighbor i):
//   0:(+1, 0)  1:(0,+1)  2:(-1,+1)  3:(-1,0)  4:(0,-1)  5:(+1,-1)

export const NEIGHBOR_DQ = [ 1,  0, -1, -1,  0,  1];
export const NEIGHBOR_DR = [ 0,  1,  1,  0, -1, -1];

export function coordKey(q: number, r: number): string { return `${q},${r}`; }

export function hexWorldPos(q: number, r: number, size: number): [number, number] {
  return [size * 1.5 * q, size * Math.sqrt(3) * (r + q * 0.5)];
}

export function hexCorners(cx: number, cy: number, size: number): [number, number][] {
  const corners: [number, number][] = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i;
    corners.push([cx + size * Math.cos(a), cy + size * Math.sin(a)]);
  }
  return corners;
}

export function hexDistance(q1: number, r1: number, q2: number, r2: number): number {
  const dq = q1 - q2, dr = r1 - r2;
  return Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr));
}

function getOrCreate(grid: HexGrid, q: number, r: number): number {
  const key = coordKey(q, r);
  let idx = grid.coordMap.get(key);
  if (idx !== undefined) return idx;
  const [wx, wy] = hexWorldPos(q, r, grid.hexSize);
  idx = grid.tiles.length;
  const tile: HexTile = { index: idx, q, r, wx, wy, isLand: false, regionId: -1 };
  grid.tiles.push(tile);
  grid.coordMap.set(key, idx);
  return idx;
}

export function generateContinent(targetLandCount: number, seed: string, hexSize: number): HexGrid {
  const grid: HexGrid = { tiles: [], coordMap: new Map(), landIndices: [], hexSize };

  const centerIdx = getOrCreate(grid, 0, 0);
  grid.tiles[centerIdx].isLand = true;
  grid.landIndices.push(centerIdx);

  const frontier: number[] = [centerIdx];
  const maxRadius = Math.ceil(Math.sqrt(targetLandCount) * 1.2);
  let iter = 0;

  while (grid.landIndices.length < targetLandCount && frontier.length > 0) {
    const fi = chooseIndex(seed, `continent|pick|${iter}`, frontier.length);
    const tileIdx = frontier[fi];
    const { q, r } = grid.tiles[tileIdx];

    const nbEdge = chooseIndex(seed, `continent|nb|${iter}`, 6);
    const nq = q + NEIGHBOR_DQ[nbEdge];
    const nr = r + NEIGHBOR_DR[nbEdge];

    iter++;
    const dist = hexDistance(0, 0, nq, nr);
    if (dist > maxRadius + 2) continue;

    const acceptChance = Math.max(0.1, Math.min(1.0, 1 - dist / maxRadius));
    if (rand01(seed, `continent|accept|${iter}`) > acceptChance) continue;

    const nbIdx = getOrCreate(grid, nq, nr);
    if (!grid.tiles[nbIdx].isLand) {
      grid.tiles[nbIdx].isLand = true;
      grid.landIndices.push(nbIdx);
      frontier.push(nbIdx);
    }

    // Remove frontier tile if all 6 neighbors are land
    let allLand = true;
    for (let e = 0; e < 6; e++) {
      const nnIdx = grid.coordMap.get(coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]));
      if (nnIdx === undefined || !grid.tiles[nnIdx].isLand) { allLand = false; break; }
    }
    if (allLand) frontier.splice(fi, 1);

    if (iter > targetLandCount * 20) break;
  }

  // Create 1-ring water border for neighbor lookups
  const waterToCreate = new Set<string>();
  for (const li of grid.landIndices) {
    const { q, r } = grid.tiles[li];
    for (let e = 0; e < 6; e++) {
      const k = coordKey(q + NEIGHBOR_DQ[e], r + NEIGHBOR_DR[e]);
      if (!grid.coordMap.has(k)) waterToCreate.add(k);
    }
  }
  for (const k of waterToCreate) {
    const [qs, rs] = k.split(',').map(Number);
    getOrCreate(grid, qs, rs); // isLand defaults to false
  }

  return grid;
}
