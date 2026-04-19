import { WorldState } from '../sim/types.js';

function fmt(n: number): string { return Math.round(n).toLocaleString(); }

export function updateHUD(world: WorldState): void {
  const tickEl = document.getElementById('hud-tick');
  const bioEl  = document.getElementById('hud-biomass');
  const spEl   = document.getElementById('hud-species');

  if (tickEl) tickEl.textContent = `Tick: ${world.currentTick}`;
  if (bioEl)  bioEl.textContent  = `Biomass: ${fmt(world.metrics.totalWorldBiomass)}`;
  if (spEl)   spEl.textContent   = `Species: ${world.metrics.aliveSpeciesCount}`;
}

export type SpeedMode = 'pause' | 'slow' | 'fast' | 'turbo';
export const SPEED_MS: Record<SpeedMode, number> = {
  pause: 0, slow: 5000, fast: 1000, turbo: 500,
};

export function initSpeedButtons(onSpeed: (mode: SpeedMode) => void): void {
  const btns: [string, SpeedMode][] = [
    ['btn-pause', 'pause'], ['btn-slow', 'slow'], ['btn-fast', 'fast'], ['btn-turbo', 'turbo'],
  ];
  for (const [id, mode] of btns) {
    document.getElementById(id)?.addEventListener('click', () => {
      setActiveSpeed(mode);
      onSpeed(mode);
    });
  }
}

export function setActiveSpeed(mode: SpeedMode): void {
  for (const id of ['btn-pause', 'btn-slow', 'btn-fast', 'btn-turbo']) {
    document.getElementById(id)?.classList.remove('active');
  }
  const modeToId: Record<SpeedMode, string> = {
    pause: 'btn-pause', slow: 'btn-slow', fast: 'btn-fast', turbo: 'btn-turbo',
  };
  document.getElementById(modeToId[mode])?.classList.add('active');
}
