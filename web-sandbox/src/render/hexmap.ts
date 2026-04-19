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

  // Background
  svg.appendChild(el('rect', { x: 0, y: 0, width: vw, height: vh, fill: '#0a1628' }));

  // Layer: hex fills
  const fillLayer = el<SVGGElement>('g');
  // Layer: region borders (static)
  const borderLayer = el<SVGGElement>('g');
  // Layer: selection highlight
  const selectionLayer = el<SVGGElement>('g');
  // Layer: migration arrows
  const arrowLayer = el<SVGGElement>('g');
  // Layer: labels
  const labelLayer = el<SVGGElement>('g');
  // Layer: click targets (invisible, on top)
  const clickLayer = el<SVGGElement>('g');

  svg.append(fillLayer, borderLayer, selectionLayer, arrowLayer, labelLayer, clickLayer);

  // Build hex fill polygons
  for (const li of grid.landIndices) {
    const tile = grid.tiles[li];
    const cx = tile.wx + ox, cy = tile.wy + oy;
    const corners = hexCorners(cx, cy, size);
    const pts = corners.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
    const biome = getBiome(regions[tile.regionId]?.biomeId ?? 'grassland');
    const poly = el<SVGPolygonElement>('polygon', {
      points: pts,
      fill: biome.color,
      stroke: biome.color,
      'stroke-width': '0.4',
      'data-tile': li,
    });
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

  // Click targets: one per region (convex polygon over all hexes, approximated as invisible rect over centroid)
  // Actually: per region, one large transparent polygon of all hex points
  for (const region of regions) {
    // Use a circle at centroid for click target — quick and reliable
    const cx = region.centroidX + ox, cy = region.centroidY + oy;
    const approxRadius = Math.sqrt(region.hexCount) * size * 0.8;
    const circle = el<SVGCircleElement>('circle', {
      cx: cx.toFixed(2), cy: cy.toFixed(2),
      r: approxRadius.toFixed(2),
      fill: 'transparent',
      'data-region': region.id,
    });
    circle.addEventListener('click', () => onRegionClick(region.id));
    circle.style.cursor = 'pointer';
    clickLayer.appendChild(circle);
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
  for (const li of region.hexIndices) {
    const tile = grid.tiles[li];
    const cx = tile.wx + ox, cy = tile.wy + oy;
    const corners = hexCorners(cx, cy, size);
    const pts = corners.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
    state.selectionLayer.appendChild(el('polygon', {
      points: pts, fill: 'none',
      stroke: '#fbbf24', 'stroke-width': '2.5',
    }));
  }
}

function renderLabels(state: HexMapState, world: WorldState, ox: number, oy: number): void {
  state.labelLayer.innerHTML = '';
  for (const region of world.regions) {
    const cx = region.centroidX + ox;
    const cy = region.centroidY + oy;
    const dominant = region.dominantSpecies;
    const spec = dominant ? getSpecies(dominant) : null;
    const trendChar = region.biomassTrend === 'up' ? '▲' : region.biomassTrend === 'down' ? '▼' : '';
    const trendColor = region.biomassTrend === 'up' ? '#34d399' : region.biomassTrend === 'down' ? '#f87171' : '#9ca3af';
    const label = spec ? spec.char : '—';

    // Dominant species abbreviation
    const txt = el<SVGTextElement>('text', {
      x: cx.toFixed(2), y: (cy + 1).toFixed(2),
      'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: spec ? spec.color : '#666',
      'font-size': '7', 'font-weight': '700',
      'pointer-events': 'none',
    });
    txt.textContent = label;
    state.labelLayer.appendChild(txt);

    if (trendChar) {
      const trendTxt = el<SVGTextElement>('text', {
        x: (cx + 6).toFixed(2), y: (cy - 3).toFixed(2),
        'text-anchor': 'middle', 'dominant-baseline': 'middle',
        fill: trendColor, 'font-size': '5.5',
        'pointer-events': 'none',
      });
      trendTxt.textContent = trendChar;
      state.labelLayer.appendChild(trendTxt);
    }
  }
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

  for (const { from, to, totalCount, topSp } of edgeMap.values()) {
    const r1 = world.regions[from], r2 = world.regions[to];
    if (!r1 || !r2) continue;

    const x1 = r1.centroidX + ox, y1 = r1.centroidY + oy;
    const x2 = r2.centroidX + ox, y2 = r2.centroidY + oy;

    // Offset arrow endpoint slightly inside target to clear the arrowhead
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len, uy = dy / len;
    const margin = world.grid.hexSize * 0.8;
    const tx = x2 - ux * margin, ty = y2 - uy * margin;

    const strokeW = Math.max(1, Math.min(4, totalCount / 6));
    const spec = SPECIES.find(s => s.id === topSp);
    const color = spec ? spec.color : '#aaa';

    state.arrowLayer.appendChild(el('line', {
      x1: x1.toFixed(2), y1: y1.toFixed(2),
      x2: tx.toFixed(2), y2: ty.toFixed(2),
      stroke: color, 'stroke-width': strokeW.toFixed(1),
      'stroke-opacity': '0.75',
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
