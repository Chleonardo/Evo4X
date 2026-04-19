import { WorldState, DEFAULT_CONFIG } from './sim/types.js';
import { generateWorld } from './sim/worldgen.js';
import { simulateTick } from './sim/simulation.js';
import { initHexMap, updateHexMap, setSelectedRegion, HexMapState } from './render/hexmap.js';
import { renderInspector, clearInspector } from './render/inspector.js';
import { renderChart } from './render/chart.js';
import { updateHUD, initSpeedButtons, SpeedMode, SPEED_MS, setActiveSpeed } from './render/hud.js';
import { saveWorld, loadWorldFromJSON } from './savegame.js';

let world: WorldState;
let mapState: HexMapState;
let selectedRegionId = -1;
let tickInterval: ReturnType<typeof setInterval> | null = null;

const mapSvg    = document.getElementById('map-svg') as unknown as SVGSVGElement;
const inspector = document.getElementById('inspector') as HTMLElement;
const chartSvg  = document.getElementById('chart-svg') as unknown as SVGSVGElement;

function onRegionClick(regionId: number): void {
  selectedRegionId = regionId === selectedRegionId ? -1 : regionId;
  setSelectedRegion(mapState, world, selectedRegionId);
  if (selectedRegionId >= 0) {
    renderInspector(inspector, world, selectedRegionId);
  } else {
    clearInspector(inspector);
  }
}

function tick(): void {
  simulateTick(world);
  updateHexMap(mapState, world);
  updateHUD(world);
  renderChart(chartSvg, world);
  if (selectedRegionId >= 0) renderInspector(inspector, world, selectedRegionId);
}

function setSpeed(mode: SpeedMode): void {
  if (tickInterval !== null) { clearInterval(tickInterval); tickInterval = null; }
  if (mode !== 'pause') {
    tickInterval = setInterval(tick, SPEED_MS[mode]);
  }
}

function buildWorld(seed: string): void {
  showLoading(true);
  // Defer to next frame so loading overlay renders first
  setTimeout(() => {
    if (tickInterval !== null) { clearInterval(tickInterval); tickInterval = null; }
    const config = { ...DEFAULT_CONFIG, worldSeed: seed };
    world = generateWorld(config);
    mapState = initHexMap(mapSvg, world, onRegionClick);
    updateHexMap(mapState, world);
    updateHUD(world);
    renderChart(chartSvg, world);
    clearInspector(inspector);
    selectedRegionId = -1;
    setActiveSpeed('pause');
    showLoading(false);
  }, 20);
}

function showLoading(show: boolean): void {
  let overlay = document.getElementById('loading');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'loading';
    overlay.textContent = 'Generating world…';
    document.body.appendChild(overlay);
  }
  overlay.classList.toggle('hidden', !show);
}

// Init
initSpeedButtons((mode) => setSpeed(mode));

document.getElementById('btn-new-world')?.addEventListener('click', () => {
  const seed = String(Math.floor(Math.random() * 999999));
  buildWorld(seed);
});

document.getElementById('btn-save')?.addEventListener('click', () => {
  if (world) saveWorld(world);
});

const loadInput = document.getElementById('load-input') as HTMLInputElement;
loadInput?.addEventListener('change', () => {
  const file = loadInput.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const json = e.target?.result as string;
      if (tickInterval !== null) { clearInterval(tickInterval); tickInterval = null; }
      showLoading(true);
      setTimeout(() => {
        world = loadWorldFromJSON(json);
        mapState = initHexMap(mapSvg, world, onRegionClick);
        updateHexMap(mapState, world);
        updateHUD(world);
        renderChart(chartSvg, world);
        clearInspector(inspector);
        selectedRegionId = -1;
        setActiveSpeed('pause');
        showLoading(false);
      }, 20);
    } catch (err) {
      alert('Failed to load save file: ' + err);
    }
    loadInput.value = '';
  };
  reader.readAsText(file);
});

buildWorld(DEFAULT_CONFIG.worldSeed);
