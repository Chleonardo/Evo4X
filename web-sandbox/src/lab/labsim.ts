import { SPECIES } from '../sim/species.js';
import { BiomeData } from '../sim/biomes.js';

export interface LabTickRecord {
  tick: number;
  populations: Record<string, number>;
  foodAllocated: Record<string, number>;
  starvation: Record<string, number>;
  totalBiomass: number;
}

export interface LabCell {
  populations: Map<string, number>;
  R1: number;
  R2: number;
  R3: number;
}

export function makeCellFromBiome(
  biome: BiomeData,
  totalResource: number,
  pops: Record<string, number>,
): LabCell {
  const sum = biome.ratioR1 + biome.ratioR2 + biome.ratioR3 || 1;
  return {
    populations: new Map(Object.entries(pops).map(([k, v]) => [k, v])),
    R1: totalResource * biome.ratioR1 / sum,
    R2: totalResource * biome.ratioR2 / sum,
    R3: totalResource * biome.ratioR3 / sum,
  };
}

export function runLab(cell: LabCell, ticks: number): LabTickRecord[] {
  const history: LabTickRecord[] = [];
  const pops = new Map(cell.populations);

  for (let t = 0; t <= ticks; t++) {
    const rec: LabTickRecord = {
      tick: t,
      populations: Object.fromEntries(pops),
      foodAllocated: {},
      starvation: {},
      totalBiomass: 0,
    };
    for (const pop of pops.values()) rec.totalBiomass += pop;
    history.push(rec);
    if (t === ticks) break;

    // births
    const preBirth = new Map<string, number>();
    for (const [sp, pop] of pops) {
      const spec = SPECIES.find(s => s.id === sp)!;
      preBirth.set(sp, pop * (1 + spec.r));
    }

    // resource allocation: w = popPreBirth * consumption * bRi
    const food = new Map<string, number>();
    for (const sp of pops.keys()) food.set(sp, 0);

    const avail = [cell.R1, cell.R2, cell.R3];
    const bW = [
      (sp: string) => SPECIES.find(s => s.id === sp)!.bR1,
      (sp: string) => SPECIES.find(s => s.id === sp)!.bR2,
      (sp: string) => SPECIES.find(s => s.id === sp)!.bR3,
    ];

    for (let ri = 0; ri < 3; ri++) {
      if (avail[ri] <= 0) continue;
      let totalW = 0;
      const ws: [string, number][] = [];
      for (const [sp, pb] of preBirth) {
        const spec = SPECIES.find(s => s.id === sp)!;
        const w = pb * spec.consumption * bW[ri](sp);
        if (w > 0) { ws.push([sp, w]); totalW += w; }
      }
      if (totalW <= 0) continue;
      for (const [sp, w] of ws) {
        food.set(sp, food.get(sp)! + avail[ri] * w / totalW);
      }
    }

    // survival: survivors = min(popPreBirth, foodAllocated / consumption)
    for (const [sp, pb] of preBirth) {
      const spec = SPECIES.find(s => s.id === sp)!;
      const f = food.get(sp)!;
      const survivors = Math.max(0, Math.min(pb, f / spec.consumption));
      rec.foodAllocated[sp] = f;
      rec.starvation[sp] = Math.max(0, pb - survivors);
      if (survivors < 0.01) pops.delete(sp);
      else pops.set(sp, survivors);
    }
  }

  return history;
}
