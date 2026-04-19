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

  // Arrowhead marker
  const defs = el<SVGDefsElement>('defs');
  const marker = el<SVGMarkerElement>('marker', {
    id: 'arrow', markerWidth: '8', markerHeight: '8',
    refX: '6', refY: '3', orient: 'auto',
  });
  const arrowPoly = el<SVGPolygonElement>('polygon', {
    points: '0 0, 6 3, 0 6', fill: 'rgba(255,255,255,0.7)',
  });
  marker.appendChild(arrowPoly);
  defs.appendChild(marker);
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

function nearestHex(hexIndices: number[], tiles: WorldState['grid']['tiles'], targetX: number, targetY: number): [number, number] {
  let nearX = 0, nearY = 0, nearDist2 = Infinity;
  for (const hi of hexIndices) {
    const t = tiles[hi];
    const dx = t.wx - targetX, dy = t.wy - targetY;
    const d2 = dx * dx + dy * dy;
    if (d2 < nearDist2) { nearDist2 = d2; nearX = t.wx; nearY = t.wy; }
  }
  return [nearX, nearY];
}

function renderArrows(state: HexMapState, world: WorldState, ox: number, oy: number): void {
  state.arrowLayer.innerHTML = '';

  // Aggregate edges by (from, to) pair — sum count across species
  const edgeMap = new Map<string, { from: number; to: number; totalCount: number; topSp: string; topCount: number }>();
  for (const edge of world.migrationEdgesThisTick) {
    const k = `${edge.fromRegion}->${edge.toRegion}`;
    const existing = edgeMap.get(k);
    if (existing) {
      existing.totalCount += edge.count;
      if (edge.count > existing.topCount) { existing.topSp = edge.speciesId; existing.topCount = edge.count; }
    } else {
      edgeMap.set(k, { from: edge.fromRegion, to: edge.toRegion, totalCount: edge.count, topSp: edge.speciesId, topCount: edge.count });
    }
  }

  const tiles = world.grid.tiles;
  const minArrowCount = 4; // hide trivial 1-2 migrant flows
  for (const { from, to, totalCount, topSp } of edgeMap.values()) {
    if (totalCount < minArrowCount) continue;
    const r1 = world.regions[from], r2 = world.regions[to];
    if (!r1 || !r2) continue;

    // Arrow from nearest hex in r1 (toward r2) to nearest hex in r2 (toward r1)
    const [sx, sy] = nearestHex(r1.hexIndices, tiles, r2.centroidX, r2.centroidY);
    const [ex, ey] = nearestHex(r2.hexIndices, tiles, r1.centroidX, r1.centroidY);
    const x1 = sx + ox, y1 = sy + oy;
    const x2 = ex + ox, y2 = ey + oy;

    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len, uy = dy / len;
    const margin = world.grid.hexSize * 0.5;
    const tx = x2 - ux * margin, ty = y2 - uy * margin;

    const strokeW = Math.max(1, Math.min(4, totalCount / 6));
    const spec = SPECIES.find(s => s.id === topSp);
    const color = spec ? spec.color : '#aaa';

    state.arrowLayer.appendChild(el('line', {
      x1: x1.toFixed(2), y1: y1.toFixed(2),
      x2: tx.toFixed(2), y2: ty.toFixed(2),
      stroke: color, 'stroke-width': strokeW.toFixed(1),
      'stroke-opacity': '0.8',
      'marker-end': 'url(#arrow)',
    }));
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
