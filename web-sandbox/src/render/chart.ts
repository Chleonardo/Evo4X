import { WorldState } from '../sim/types.js';
import { SPECIES } from '../sim/species.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const W = 280, H = 140, PAD = { top: 8, right: 8, bottom: 20, left: 36 };
const CHART_W = W - PAD.left - PAD.right;
const CHART_H = H - PAD.top - PAD.bottom;
const WINDOW = 80;

function el<T extends SVGElement>(tag: string, attrs: Record<string, string | number> = {}): T {
  const e = document.createElementNS(SVG_NS, tag) as T;
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

export function renderChart(svg: SVGSVGElement, world: WorldState): void {
  svg.innerHTML = '';
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const histories = new Map<string, number[]>();
  for (const sp of SPECIES) {
    const h = world.globalPopHistory.get(sp.id);
    if (h && h.length > 0) histories.set(sp.id, h);
  }
  if (histories.size === 0) return;

  // Find max across all series and time window
  let maxPop = 1;
  for (const [, hist] of histories) {
    const slice = hist.slice(-WINDOW);
    for (const v of slice) if (v > maxPop) maxPop = v;
  }

  const gx = PAD.left, gy = PAD.top;

  // Background
  svg.appendChild(el('rect', { x: gx, y: gy, width: CHART_W, height: CHART_H, fill: '#111827', rx: 3 }));

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const y = gy + CHART_H * (1 - i / 4);
    svg.appendChild(el('line', { x1: gx, y1: y.toFixed(1), x2: gx + CHART_W, y2: y.toFixed(1), stroke: '#1f2937', 'stroke-width': 0.8 }));
    const label = Math.round(maxPop * i / 4);
    const txt = el<SVGTextElement>('text', { x: gx - 3, y: y.toFixed(1), 'text-anchor': 'end', 'dominant-baseline': 'middle', fill: '#4b5563', 'font-size': 7 });
    txt.textContent = label >= 1000 ? `${Math.round(label / 1000)}k` : String(label);
    svg.appendChild(txt);
  }

  // X axis label
  const maxTick = world.currentTick;
  const minTick = Math.max(0, maxTick - WINDOW);
  (() => {
    const t = el<SVGTextElement>('text', { x: gx, y: gy + CHART_H + 14, fill: '#4b5563', 'font-size': 7 });
    t.textContent = `t=${minTick}`;
    svg.appendChild(t);
    const t2 = el<SVGTextElement>('text', { x: gx + CHART_W, y: gy + CHART_H + 14, 'text-anchor': 'end', fill: '#4b5563', 'font-size': 7 });
    t2.textContent = `t=${maxTick}`;
    svg.appendChild(t2);
  })();

  // Lines per species
  for (const sp of SPECIES) {
    const hist = histories.get(sp.id);
    if (!hist || hist.length < 2) continue;
    const slice = hist.slice(-WINDOW);
    const n = slice.length;
    const pts = slice.map((v, i) => {
      const x = gx + (i / (WINDOW - 1)) * CHART_W;
      const y = gy + CHART_H * (1 - v / maxPop);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    svg.appendChild(el('polyline', {
      points: pts, fill: 'none', stroke: sp.color,
      'stroke-width': 1.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }));
    // Dot at last point
    const lastX = gx + ((n - 1) / (WINDOW - 1)) * CHART_W;
    const lastY = gy + CHART_H * (1 - slice[n - 1] / maxPop);
    svg.appendChild(el('circle', { cx: lastX.toFixed(1), cy: lastY.toFixed(1), r: 2, fill: sp.color }));
  }
}
