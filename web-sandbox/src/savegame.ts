import { WorldState, SimConfig } from './sim/types.js';
import { generateWorld } from './sim/worldgen.js';
import { updateMetrics } from './sim/worldgen.js';

interface RegionSave {
  populations: Record<string, number>;
  activeEffects: { effectType: string; multR1: number; multR2: number; multR3: number; ttl: number }[];
  popHistory: Record<string, number[]>;
}

interface SaveFile {
  version: 1;
  config: SimConfig;
  tick: number;
  regions: RegionSave[];
  globalPopHistory: Record<string, number[]>;
}

export function saveWorld(world: WorldState): void {
  const save: SaveFile = {
    version: 1,
    config: world.config,
    tick: world.currentTick,
    regions: world.regions.map(r => ({
      populations: Object.fromEntries(r.populations),
      activeEffects: r.activeEffects.map(e => ({ ...e })),
      popHistory: Object.fromEntries([...r.popHistory].map(([k, v]) => [k, [...v]])),
    })),
    globalPopHistory: Object.fromEntries([...world.globalPopHistory].map(([k, v]) => [k, [...v]])),
  };
  const blob = new Blob([JSON.stringify(save, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `evo4x_seed${world.config.worldSeed}_t${world.currentTick}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function loadWorldFromJSON(json: string): WorldState {
  const save: SaveFile = JSON.parse(json);
  const world = generateWorld(save.config);
  world.currentTick = save.tick;

  for (let i = 0; i < save.regions.length && i < world.regions.length; i++) {
    const r = world.regions[i];
    const s = save.regions[i];
    r.populations = new Map(Object.entries(s.populations).map(([k, v]) => [k, Number(v)]));
    r.activeEffects = s.activeEffects;
    r.popHistory = new Map(Object.entries(s.popHistory));
  }
  world.globalPopHistory = new Map(Object.entries(save.globalPopHistory));
  updateMetrics(world);
  return world;
}
