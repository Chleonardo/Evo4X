export interface SpeciesData {
  id: string;
  name: string;
  char: string;
  emoji: string;
  r: number;
  bR1: number;
  bR2: number;
  bR3: number;
  radiation: number; // 0-100 %
  color: string;
}

export const SPECIES: SpeciesData[] = [
  { id: 'gazelle',  name: 'Gazelle',  char: 'Gz', emoji: '🦌', r: 1.00, bR1: 1.0, bR2: 0.0, bR3: 0.0, radiation:  8, color: '#e8c840' },
  { id: 'impala',   name: 'Impala',   char: 'Im', emoji: '🐐', r: 0.95, bR1: 1.2, bR2: 0.6, bR3: 0.0, radiation:  6, color: '#c07830' },
  { id: 'zebra',    name: 'Zebra',    char: 'Zb', emoji: '🦓', r: 0.75, bR1: 1.1, bR2: 0.0, bR3: 0.0, radiation:  5, color: '#c8c8c8' },
  { id: 'buffalo',  name: 'Buffalo',  char: 'Bf', emoji: '🐃', r: 0.65, bR1: 2.0, bR2: 0.0, bR3: 0.0, radiation:  3, color: '#6a4020' },
  { id: 'giraffe',  name: 'Giraffe',  char: 'Gi', emoji: '🦒', r: 0.40, bR1: 0.5, bR2: 3.0, bR3: 0.0, radiation:  4, color: '#d8bc40' },
  { id: 'elephant', name: 'Elephant', char: 'El', emoji: '🐘', r: 0.12, bR1: 2.0, bR2: 3.0, bR3: 0.0, radiation:  2, color: '#888888' },
  { id: 'warthog',  name: 'Warthog',  char: 'Wh', emoji: '🐗', r: 1.20, bR1: 0.5, bR2: 0.0, bR3: 2.0, radiation:  6, color: '#a86040' },
  { id: 'mole_rat', name: 'Mole Rat', char: 'Mr', emoji: '🐀', r: 1.80, bR1: 0.0, bR2: 0.0, bR3: 1.0, radiation:  5, color: '#b89060' },
];

const SPECIES_MAP = new Map<string, SpeciesData>(SPECIES.map(s => [s.id, s]));

export function getSpecies(id: string): SpeciesData {
  const s = SPECIES_MAP.get(id);
  if (!s) throw new Error(`Unknown species: ${id}`);
  return s;
}
