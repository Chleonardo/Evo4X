import { BIOMES, BiomeData } from './sim/biomes.js';
import { SPECIES } from './sim/species.js';
import { makeCellFromBiome, runLab, LabTickRecord } from './lab/labsim.js';

const TICKS = 80;
const DEFAULT_RESOURCE = 200; // ~20 hexes × richnessPerHex=10

interface TestCase {
  name: string;
  pops: Record<string, number>;
  resourceScale?: number;
}

const TEST_CASES: TestCase[] = [
  { name: 'Mono: Gazelle ×100',   pops: { gazelle: 100 } },
  { name: 'Mono: Zebra ×100',     pops: { zebra: 100 } },
  { name: 'Mono: Elephant ×100',  pops: { elephant: 100 } },
  { name: 'Mixed Grassland',       pops: { gazelle: 100, zebra: 50, buffalo: 20 } },
  { name: 'Mixed Leaves',          pops: { impala: 50, giraffe: 30, elephant: 10 } },
  { name: 'Root Heavy',            pops: { warthog: 80, mole_rat: 150 } },
  { name: 'Diverse Mix',           pops: { gazelle: 50, impala: 30, zebra: 30, buffalo: 15, giraffe: 20, elephant: 10, warthog: 20, mole_rat: 60 } },
  { name: 'Stress (low res)',      pops: { gazelle: 100, zebra: 50, buffalo: 20 }, resourceScale: 0.3 },
];

// ── SVG helpers ──────────────────────────────────────────────────────────────

const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl<T extends SVGElement>(tag: string, attrs: Record<string, string | number> = {}): T {
  const e = document.createElementNS(SVG_NS, tag) as T;
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

const W = 320, H = 200;
const PAD = { t: 8, r: 8, b: 24, l: 36 };
const CW = W - PAD.l - PAD.r;
const CH = H - PAD.t - PAD.b;

function drawChart(svg: SVGSVGElement, history: LabTickRecord[], species: string[], title: string): void {
  svg.innerHTML = '';
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', String(W));
  svg.setAttribute('height', String(H));

  const maxPop = Math.max(
    10,
    ...history.flatMap(r => species.map(sp => r.populations[sp] ?? 0))
  ) * 1.05;
  const maxTick = history.length - 1 || 1;

  const tx = (t: number) => PAD.l + (t / maxTick) * CW;
  const ty = (p: number) => PAD.t + CH - (p / maxPop) * CH;

  // bg
  svg.appendChild(svgEl('rect', { x: 0, y: 0, width: W, height: H, fill: '#1a1a2e' }));
  svg.appendChild(svgEl('rect', { x: PAD.l, y: PAD.t, width: CW, height: CH, fill: '#0f0f1a' }));

  // grid lines
  for (let i = 0; i <= 4; i++) {
    const yv = PAD.t + (i / 4) * CH;
    svg.appendChild(svgEl('line', { x1: PAD.l, y1: yv, x2: PAD.l + CW, y2: yv, stroke: '#333', 'stroke-width': 0.5 }));
    const label = ((1 - i / 4) * maxPop).toFixed(0);
    const t = svgEl<SVGTextElement>('text', { x: PAD.l - 3, y: yv, 'text-anchor': 'end', 'dominant-baseline': 'middle', 'font-size': 8, fill: '#888' });
    t.textContent = label;
    svg.appendChild(t);
  }

  // x-axis ticks
  for (let i = 0; i <= 4; i++) {
    const xv = PAD.l + (i / 4) * CW;
    const label = String(Math.round((i / 4) * maxTick));
    const t = svgEl<SVGTextElement>('text', { x: xv, y: PAD.t + CH + 9, 'text-anchor': 'middle', 'font-size': 8, fill: '#888' });
    t.textContent = label;
    svg.appendChild(t);
  }

  // species lines
  for (const sp of species) {
    const spec = SPECIES.find(s => s.id === sp);
    if (!spec) continue;
    const pts = history.map((r, i) => `${tx(i).toFixed(1)},${ty(r.populations[sp] ?? 0).toFixed(1)}`).join(' ');
    svg.appendChild(svgEl('polyline', { points: pts, fill: 'none', stroke: spec.color, 'stroke-width': 1.5, 'stroke-linejoin': 'round' }));
  }

  // title
  const tt = svgEl<SVGTextElement>('text', { x: PAD.l + CW / 2, y: PAD.t - 1, 'text-anchor': 'middle', 'font-size': 9, fill: '#ccc' });
  tt.textContent = title;
  svg.appendChild(tt);
}

function drawLegend(container: HTMLElement, species: string[]): void {
  container.innerHTML = '';
  for (const sp of species) {
    const spec = SPECIES.find(s => s.id === sp);
    if (!spec) continue;
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-dot" style="background:${spec.color}"></span>${spec.emoji} ${spec.name}`;
    container.appendChild(item);
  }
}

// ── Render all ───────────────────────────────────────────────────────────────

let activeBiome: BiomeData = BIOMES[0];

function renderAll(): void {
  const grid = document.getElementById('chart-grid')!;
  grid.innerHTML = '';

  for (const tc of TEST_CASES) {
    const species = Object.keys(tc.pops);
    const total = DEFAULT_RESOURCE * (tc.resourceScale ?? 1);
    const cell = makeCellFromBiome(activeBiome, total, tc.pops);
    const history = runLab(cell, TICKS);

    const card = document.createElement('div');
    card.className = 'chart-card';

    const titleEl = document.createElement('div');
    titleEl.className = 'card-title';
    titleEl.textContent = tc.name + (tc.resourceScale ? ` (res×${tc.resourceScale})` : '');
    card.appendChild(titleEl);

    const svg = document.createElementNS(SVG_NS, 'svg') as SVGSVGElement;
    drawChart(svg, history, species, '');
    card.appendChild(svg);

    const legend = document.createElement('div');
    legend.className = 'chart-legend';
    drawLegend(legend, species);
    card.appendChild(legend);

    // final pop table
    const last = history[history.length - 1];
    const table = document.createElement('div');
    table.className = 'pop-table';
    for (const sp of species) {
      const spec = SPECIES.find(s => s.id === sp)!;
      const initial = tc.pops[sp];
      const final = last.populations[sp] ?? 0;
      const row = document.createElement('div');
      row.className = 'pop-row';
      row.innerHTML = `<span style="color:${spec.color}">${spec.emoji} ${spec.name}</span>`
        + `<span>${initial} → <b>${final.toFixed(1)}</b></span>`;
      table.appendChild(row);
    }
    card.appendChild(table);

    grid.appendChild(card);
  }
}

// ── Biome selector ───────────────────────────────────────────────────────────

function buildBiomeButtons(): void {
  const bar = document.getElementById('biome-bar')!;
  for (const biome of BIOMES) {
    const btn = document.createElement('button');
    btn.className = 'biome-btn' + (biome === activeBiome ? ' active' : '');
    btn.style.borderColor = biome.color;
    btn.textContent = biome.name;
    btn.addEventListener('click', () => {
      activeBiome = biome;
      bar.querySelectorAll('.biome-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderAll();
    });
    bar.appendChild(btn);
  }
}

buildBiomeButtons();
renderAll();
