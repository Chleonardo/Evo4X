import { WorldState } from '../sim/types.js';
import { getBiome } from '../sim/biomes.js';
import { getSpecies } from '../sim/species.js';

function fmt(n: number): string { return Math.round(n).toLocaleString(); }

export function renderInspector(container: HTMLElement, world: WorldState, regionId: number): void {
  const region = world.regions[regionId];
  if (!region) { container.innerHTML = '<div class="inspector-empty">Region not found</div>'; return; }

  const biome = getBiome(region.biomeId);

  // Sort species by population descending
  const species = [...region.populations.entries()].sort((a, b) => b[1] - a[1]);
  const breakdown = region.tickBreakdown;

  const trendHTML = (sp: string) => {
    const bd = breakdown.get(sp);
    if (!bd) return '';
    return bd.trend === 'up' ? '<span class="trend-up">▲</span>'
         : bd.trend === 'down' ? '<span class="trend-down">▼</span>'
         : '<span class="trend-flat">—</span>';
  };

  const speciesRows = species.map(([sp, pop]) => {
    const spec = getSpecies(sp);
    const bd = breakdown.get(sp);
    const migIn  = bd ? fmt(bd.incomingMigrants) : '0';
    const migOut = bd ? fmt(bd.outgoingMigrants) : '0';
    return `
      <div class="insp-species-row">
        <span class="insp-species-emoji">${spec.emoji}</span>
        <span class="insp-species-name">${spec.name}</span>
        <span class="insp-species-pop">${fmt(pop)}</span>
        ${trendHTML(sp)}
      </div>
      <div class="insp-mig-row" style="padding-left:14px">↓${migIn} in &nbsp; ↑${migOut} out</div>`;
  }).join('');

  container.innerHTML = `
    <div class="insp-region-name">Region ${region.id}</div>
    <div class="insp-biome-tag" style="background:${biome.color}">${biome.name}</div>
    <div class="insp-row"><span>Hexes</span><span>${region.hexCount}</span></div>
    <div class="insp-row"><span>Biomass</span><span>${fmt(region.totalBiomass)}</span></div>
    <div class="insp-section-title">Resources (effective)</div>
    <div class="insp-row"><span>🌱 Grass</span><span>${fmt(region.effR1)}</span></div>
    <div class="insp-row"><span>🌿 Leaves</span><span>${fmt(region.effR2)}</span></div>
    <div class="insp-row"><span>🥔 Roots</span><span>${fmt(region.effR3)}</span></div>
    <div class="insp-section-title">Species (${species.length})</div>
    ${speciesRows || '<div class="insp-mig-row">No species</div>'}
  `;
}

export function clearInspector(container: HTMLElement): void {
  container.innerHTML = '<div class="inspector-empty">Click a region to inspect</div>';
}
