import { BIOMES, BiomeData } from './sim/biomes.js';
import { SPECIES } from './sim/species.js';
import { makeCellFromBiome, runLab, LabTickRecord, LAB_CONSUMPTION } from './lab/labsim.js';

const TICKS = 80;
const TOTAL_RESOURCE = 200;
const POP_PER_SPECIES = 50;

// ── SVG helpers ──────────────────────────────────────────────────────────────

const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl<T extends SVGElement>(tag: string, attrs: Record<string, string | number> = {}): T {
  const e = document.createElementNS(SVG_NS, tag) as T;
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

const W = 300, H = 190;
const PAD = { t: 8, r: 8, b: 22, l: 36 };
const CW = W - PAD.l - PAD.r;
const CH = H - PAD.t - PAD.b;

function drawChart(svg: SVGSVGElement, history: LabTickRecord[], species: string[]): void {
  svg.innerHTML = '';
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', String(W));
  svg.setAttribute('height', String(H));

  const maxPop = Math.max(10, ...history.flatMap(r => species.map(sp => r.populations[sp] ?? 0))) * 1.05;
  const maxTick = history.length - 1 || 1;

  const tx = (t: number) => PAD.l + (t / maxTick) * CW;
  const ty = (p: number) => PAD.t + CH - (p / maxPop) * CH;

  svg.appendChild(svgEl('rect', { x: 0, y: 0, width: W, height: H, fill: '#12121e' }));
  svg.appendChild(svgEl('rect', { x: PAD.l, y: PAD.t, width: CW, height: CH, fill: '#0a0a14' }));

  for (let i = 0; i <= 4; i++) {
    const yv = PAD.t + (i / 4) * CH;
    svg.appendChild(svgEl('line', { x1: PAD.l, y1: yv, x2: PAD.l + CW, y2: yv, stroke: '#2a2a3a', 'stroke-width': 0.5 }));
    const t = svgEl<SVGTextElement>('text', { x: PAD.l - 3, y: yv, 'text-anchor': 'end', 'dominant-baseline': 'middle', 'font-size': 8, fill: '#666' });
    t.textContent = ((1 - i / 4) * maxPop).toFixed(0);
    svg.appendChild(t);
  }
  for (let i = 0; i <= 4; i++) {
    const xv = PAD.l + (i / 4) * CW;
    const t = svgEl<SVGTextElement>('text', { x: xv, y: PAD.t + CH + 9, 'text-anchor': 'middle', 'font-size': 8, fill: '#666' });
    t.textContent = String(Math.round((i / 4) * maxTick));
    svg.appendChild(t);
  }

  for (const sp of species) {
    const spec = SPECIES.find(s => s.id === sp);
    if (!spec) continue;
    const pts = history.map((r, i) => `${tx(i).toFixed(1)},${ty(r.populations[sp] ?? 0).toFixed(1)}`).join(' ');
    svg.appendChild(svgEl('polyline', { points: pts, fill: 'none', stroke: spec.color, 'stroke-width': 1.5, 'stroke-linejoin': 'round' }));
  }
}

// ── Card builder ─────────────────────────────────────────────────────────────

function buildCard(biome: BiomeData, species: string[]): HTMLElement {
  const pops: Record<string, number> = {};
  for (const sp of species) pops[sp] = POP_PER_SPECIES;

  const cell = makeCellFromBiome(biome, TOTAL_RESOURCE, pops);
  const history = runLab(cell, TICKS);
  const last = history[history.length - 1];

  const card = document.createElement('div');
  card.className = 'chart-card';

  // title: species emojis
  const titleEl = document.createElement('div');
  titleEl.className = 'card-title';
  titleEl.textContent = species.map(sp => {
    const s = SPECIES.find(x => x.id === sp);
    return s ? `${s.emoji} ${s.name}` : sp;
  }).join(' + ');
  card.appendChild(titleEl);

  const svg = document.createElementNS(SVG_NS, 'svg') as SVGSVGElement;
  drawChart(svg, history, species);
  card.appendChild(svg);

  // legend + final pop
  const legendEl = document.createElement('div');
  legendEl.className = 'chart-legend';
  for (const sp of species) {
    const spec = SPECIES.find(s => s.id === sp);
    if (!spec) continue;
    const final = last.populations[sp] ?? 0;
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.innerHTML =
      `<span class="legend-dot" style="background:${spec.color}"></span>` +
      `${spec.emoji} <b style="color:${spec.color}">${final.toFixed(0)}</b>` +
      `<span class="cons-badge">×${LAB_CONSUMPTION[sp] ?? 1}</span>`;
    legendEl.appendChild(item);
  }
  card.appendChild(legendEl);

  return card;
}

// ── Render biome ─────────────────────────────────────────────────────────────

let activeBiome: BiomeData = BIOMES[0];

function renderBiome(biome: BiomeData): void {
  const grid = document.getElementById('chart-grid')!;
  grid.innerHTML = '';

  // Deduplicate packs (same species set regardless of order)
  const seen = new Set<string>();
  for (const pack of biome.packs) {
    if (pack.length === 0) continue;
    const key = [...pack].sort().join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    grid.appendChild(buildCard(biome, pack));
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
      renderBiome(biome);
    });
    bar.appendChild(btn);
  }
}

buildBiomeButtons();
renderBiome(activeBiome);
