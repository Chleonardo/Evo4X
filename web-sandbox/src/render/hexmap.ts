import { WorldState } from '../sim/types.js';
import { hexCorners, NEIGHBOR_DQ, NEIGHBOR_DR, coordKey } from '../sim/hexgrid.js';
import { getBiome } from '../sim/biomes.js';
import { getSpecies, SPECIES } from '../sim/species.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

function el<T extends SVGElement>(tag: string, attrs: Record<string, string | number> = {}): T {
  const e = document.createElementNS(SVG_NS, tag) as T;
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

export interface HexMapState {
  svg: SVGSVGElement;
  selectedRegionId: number;
  onRegionClick: (regionId: number) => void;
  arrowLayer: SVGGElement;
  labelLayer: SVGGElement;
  selectionLayer: SVGGElement;
}

export function initHexMap(svg: SVGSVGElement, world: WorldState, onRegionClick: (id: number) => void): HexMapState {
  svg.innerHTML = '';

  const { grid, regions } = world;
  const size = grid.hexSize;

  // Compute bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const li of grid.landIndices) {
    const { wx, wy } = grid.tiles[li];
    const pad = size;
    if (wx - pad < minX) minX = wx - pad;
    if (wx + pad > maxX) maxX = wx + pad;
    if (wy - pad < minY) minY = wy - pad;
    if (wy + pad > maxY) maxY = wy + pad;
  }
  const vw = maxX - minX + size * 2;
  const vh = maxY - minY + size * 2;
  const ox = -minX + size;
  const oy = -minY + size;

  svg.setAttribute('viewBox', `0 0 ${vw} ${vh}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  const defs = el<SVGDefsElement>('defs');
  svg.appendChild(defs);

  // Background (water / lake color)
  svg.appendChild(el('rect', { x: 0, y: 0, width: vw, height: vh, fill: '#1e6091' }));

  // Layer: hex fills (also serves as click targets)
  const fillLayer = el<SVGGElement>('g');
  // Layer: region borders (static, no pointer events)
  const borderLayer = el<SVGGElement>('g', { 'pointer-events': 'none' });
  // Layer: selection highlight (no pointer events)
  const selectionLayer = el<SVGGElement>('g', { 'pointer-events': 'none' });
  // Layer: migration arrows (no pointer events)
  const arrowLayer = el<SVGGElement>('g', { 'pointer-events': 'none' });
  // Layer: labels (no pointer events)
  const labelLayer = el<SVGGElement>('g', { 'pointer-events': 'none' });

  svg.append(fillLayer, borderLayer, selectionLayer, arrowLayer, labelLayer);

  // Build hex fill polygons — onclick directly on each tile poly (pixel-perfect)
  for (const li of grid.landIndices) {
    const tile = grid.tiles[li];
    const cx = tile.wx + ox, cy = tile.wy + oy;
    const corners = hexCorners(cx, cy, size);
    const pts = corners.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
    const regionId = tile.regionId;
    const biome = getBiome(regions[regionId]?.biomeId ?? 'grassland');
    const poly = el<SVGPolygonElement>('polygon', {
      points: pts,
      fill: biome.color,
      stroke: biome.color,
      'stroke-width': '0.4',
    });
    poly.style.cursor = 'pointer';
    poly.addEventListener('click', () => onRegionClick(regionId));
    fillLayer.appendChild(poly);
  }

  // Build border lines between different regions (and land-ocean edges)
  for (const li of grid.landIndices) {
    const tile = grid.tiles[li];
    const cx = tile.wx + ox, cy = tile.wy + oy;
    const corners = hexCorners(cx, cy, size);
    for (let e = 0; e < 6; e++) {
      const nbKey = coordKey(tile.q + NEIGHBOR_DQ[e], tile.r + NEIGHBOR_DR[e]);
      const nbIdx = grid.coordMap.get(nbKey);
      const sameRegion = nbIdx !== undefined && grid.tiles[nbIdx].isLand && grid.tiles[nbIdx].regionId === tile.regionId;
      if (!sameRegion) {
        const [c0, c1] = [corners[e], corners[(e + 1) % 6]];
        borderLayer.appendChild(el('line', {
          x1: c0[0].toFixed(2), y1: c0[1].toFixed(2),
          x2: c1[0].toFixed(2), y2: c1[1].toFixed(2),
          stroke: '#111', 'stroke-width': '2', 'stroke-linecap': 'round',
        }));
      }
    }
  }

  const state: HexMapState = {
    svg, selectedRegionId: -1, onRegionClick, arrowLayer, labelLayer, selectionLayer,
  };

  renderLabels(state, world, ox, oy);
  return state;
}

export function updateHexMap(state: HexMapState, world: WorldState): void {
  const { grid } = world;
  const size = grid.hexSize;
  let minX = Infinity, minY = Infinity;
  for (const li of grid.landIndices) {
    const { wx, wy } = grid.tiles[li];
    if (wx - size < minX) minX = wx - size;
    if (wy - size < minY) minY = wy - size;
  }
  const _ox = -minX + size;
  const _oy = -minY + size;

  // Update hex fill colors (biome = static, but update selection highlight)
  updateSelectionHighlight(state, world, _ox, _oy);
  renderLabels(state, world, _ox, _oy);
  renderArrows(state, world, _ox, _oy);
}

function updateSelectionHighlight(state: HexMapState, world: WorldState, ox: number, oy: number): void {
  state.selectionLayer.innerHTML = '';
  if (state.selectedRegionId < 0) return;
  const region = world.regions[state.selectedRegionId];
  if (!region) return;
  const { grid } = world;
  const size = grid.hexSize;
  const rid = state.selectedRegionId;
  // Draw only the outer border edges (not internal hex edges)
  for (const li of region.hexIndices) {
    const tile = grid.tiles[li];
    const cx = tile.wx + ox, cy = tile.wy + oy;
    const corners = hexCorners(cx, cy, size);
    for (let e = 0; e < 6; e++) {
      const nbKey = coordKey(tile.q + NEIGHBOR_DQ[e], tile.r + NEIGHBOR_DR[e]);
      const nbIdx = grid.coordMap.get(nbKey);
      const sameRegion = nbIdx !== undefined && grid.tiles[nbIdx].isLand && grid.tiles[nbIdx].regionId === rid;
      if (!sameRegion) {
        const [c0, c1] = [corners[e], corners[(e + 1) % 6]];
        state.selectionLayer.appendChild(el('line', {
          x1: c0[0].toFixed(2), y1: c0[1].toFixed(2),
          x2: c1[0].toFixed(2), y2: c1[1].toFixed(2),
          stroke: '#fbbf24', 'stroke-width': '3', 'stroke-linecap': 'round',
        }));
      }
    }
  }
}

function renderLabels(state: HexMapState, world: WorldState, ox: number, oy: number): void {
  state.labelLayer.innerHTML = '';
  for (const region of world.regions) {
    const cx = region.centroidX + ox;
    const cy = region.centroidY + oy;

    // All species with >5% of total regional biomass
    const totalBiomass = [...region.populations.values()].reduce((a, b) => a + b, 0);
    if (totalBiomass <= 0) continue;
    const significant = [...region.populations.entries()]
      .filter(([, pop]) => pop / totalBiomass >= 0.05)
      .sort((a, b) => b[1] - a[1]);

    const n = significant.length;
    const spacing = 10;
    const startX = cx - ((n - 1) * spacing) / 2;

    significant.forEach(([sp], i) => {
      const spec = getSpecies(sp);
      const txt = el<SVGTextElement>('text', {
        x: (startX + i * spacing).toFixed(2), y: cy.toFixed(2),
        'text-anchor': 'middle', 'dominant-baseline': 'middle',
        'font-size': '11', 'pointer-events': 'none',
      });
      txt.textContent = spec.emoji;
      state.labelLayer.appendChild(txt);
    });
  }
}

// Finds the hex in `region` that borders `targetRegionId`, closest to target centroid
function borderHexToward(
  region: { hexIndices: number[] },
  targetRegionId: number,
  grid: WorldState['grid'],
  targetCX: number, targetCY: number,
): [number, number] | null {
  let bestX = 0, bestY = 0, bestDist2 = Infinity, found = false;
  for (const hi of region.hexIndices) {
    const tile = grid.tiles[hi];
    let borders = false;
    for (let e = 0; e < 6; e++) {
      const ni = grid.coordMap.get(coordKey(tile.q + NEIGHBOR_DQ[e], tile.r + NEIGHBOR_DR[e]));
      if (ni !== undefined && grid.tiles[ni].regionId === targetRegionId) { borders = true; break; }
    }
    if (!borders) continue;
    const dx = tile.wx - targetCX, dy = tile.wy - targetCY;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDist2) { bestDist2 = d2; bestX = tile.wx; bestY = tile.wy; found = true; }
  }
  return found ? [bestX, bestY] : null;
}

function renderArrows(state: HexMapState, world: WorldState, ox: number, oy: number): void {
  state.arrowLayer.innerHTML = '';

  // Aggregate smoothed edges by (from, to) pair
  const edgeMap = new Map<string, { from: number; to: number; totalCount: number; topSp: string; topCount: number }>();
  for (const edge of world.migrationSmoothed.values()) {
    const k = `${edge.fromRegion}->${edge.toRegion}`;
    const existing = edgeMap.get(k);
    if (existing) {
      existing.totalCount += edge.count;
      if (edge.count > existing.topCount) { existing.topSp = edge.speciesId; existing.topCount = edge.count; }
    } else {
      edgeMap.set(k, { from: edge.fromRegion, to: edge.toRegion, totalCount: edge.count, topSp: edge.speciesId, topCount: edge.count });
    }
  }

  for (const { from, to, totalCount, topSp } of edgeMap.values()) {
    const r1 = world.regions[from], r2 = world.regions[to];
    if (!r1 || !r2) continue;

    // Arrow stays inside r1: centroid → border hex of r1 that touches r2
    const border = borderHexToward(r1, to, world.grid, r2.centroidX, r2.centroidY);
    if (!border) continue;

    const x1 = r1.centroidX + ox, y1 = r1.centroidY + oy;
    const x2 = border[0] + ox,    y2 = border[1] + oy;
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len, uy = dy / len;
    // Pull tip back slightly so arrowhead doesn't sit exactly on the border line
    const margin = world.grid.hexSize * 0.2;
    const tx = x2 - ux * margin, ty = y2 - uy * margin;

    const strokeW = Math.max(1, Math.min(4, totalCount / 5));
    const opacity = Math.min(0.9, Math.max(0, totalCount / 22));
    if (opacity < 0.05) continue;

    const spec = SPECIES.find(s => s.id === topSp);
    const color = spec ? spec.color : '#aaa';

    const headLen = 5, headW = 2.5;
    const bx = tx - ux * headLen, by = ty - uy * headLen;
    const w1x = bx - uy * headW, w1y = by + ux * headW;
    const w2x = bx + uy * headW, w2y = by - ux * headW;

    state.arrowLayer.appendChild(el('line', {
      x1: x1.toFixed(2), y1: y1.toFixed(2),
      x2: bx.toFixed(2), y2: by.toFixed(2),
      stroke: color, 'stroke-width': strokeW.toFixed(1),
      'stroke-opacity': opacity.toFixed(2),
    }));
    state.arrowLayer.appendChild(el('polygon', {
      points: `${tx.toFixed(2)},${ty.toFixed(2)} ${w1x.toFixed(2)},${w1y.toFixed(2)} ${w2x.toFixed(2)},${w2y.toFixed(2)}`,
      fill: color, 'fill-opacity': opacity.toFixed(2),
    }));

    if (spec) {
      const lx = (x1 + tx) * 0.5 - uy * 5;
      const ly = (y1 + ty) * 0.5 + ux * 5;
      const lbl = el<SVGTextElement>('text', {
        x: lx.toFixed(2), y: ly.toFixed(2),
        'text-anchor': 'middle', 'dominant-baseline': 'middle',
        'font-size': '7', 'pointer-events': 'none',
        stroke: '#000', 'stroke-width': '2', 'paint-order': 'stroke fill',
        opacity: opacity.toFixed(2),
      });
      lbl.textContent = spec.emoji;
      state.arrowLayer.appendChild(lbl);
    }
  }
}

export function setSelectedRegion(state: HexMapState, world: WorldState, regionId: number): void {
  state.selectedRegionId = regionId;
  const size = world.grid.hexSize;
  let minX = Infinity, minY = Infinity;
  for (const li of world.grid.landIndices) {
    const { wx, wy } = world.grid.tiles[li];
    if (wx - size < minX) minX = wx - size;
    if (wy - size < minY) minY = wy - size;
  }
  const ox = -minX + size, oy = -minY + size;
  updateSelectionHighlight(state, world, ox, oy);
}
