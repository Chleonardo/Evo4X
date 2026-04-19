export interface BiomeData {
  id: string;
  name: string;
  ratioR1: number; // 0-100
  ratioR2: number;
  ratioR3: number;
  color: string;  // CSS hex color
  packs: string[][];
}

export const BIOMES: BiomeData[] = [
  {
    id: 'grassland', name: 'Grassland',
    ratioR1: 100, ratioR2: 0, ratioR3: 0,
    color: '#7ed42e',
    packs: [[], ['gazelle'], ['zebra'], ['buffalo']],
  },
  {
    id: 'open_savanna', name: 'Open Savanna',
    ratioR1: 67, ratioR2: 33, ratioR3: 0,
    color: '#d4b428',
    packs: [
      ['giraffe'], ['giraffe', 'gazelle'], ['giraffe', 'zebra'],
      ['giraffe', 'buffalo'], ['giraffe', 'warthog'], ['giraffe', 'impala'], ['impala'],
    ],
  },
  {
    id: 'woodland', name: 'Woodland',
    ratioR1: 33, ratioR2: 67, ratioR3: 0,
    color: '#287828',
    packs: [['giraffe'], ['giraffe', 'buffalo'], ['giraffe', 'impala']],
  },
  {
    id: 'dense_grove', name: 'Dense Grove',
    ratioR1: 20, ratioR2: 80, ratioR3: 0,
    color: '#1a4a1a',
    packs: [['giraffe'], ['elephant']],
  },
  {
    id: 'root_patch', name: 'Root Patch',
    ratioR1: 67, ratioR2: 0, ratioR3: 33,
    color: '#c49020',
    packs: [
      ['mole_rat'], ['mole_rat', 'gazelle'], ['mole_rat', 'zebra'],
      ['mole_rat', 'buffalo'], ['mole_rat', 'giraffe'],
      ['warthog'], ['warthog', 'gazelle'], ['warthog', 'impala'],
      ['warthog', 'buffalo'], ['warthog', 'elephant'],
    ],
  },
  {
    id: 'rootland', name: 'Rootland',
    ratioR1: 33, ratioR2: 0, ratioR3: 67,
    color: '#a06018',
    packs: [
      ['mole_rat'], ['mole_rat', 'gazelle'], ['mole_rat', 'zebra'],
      ['mole_rat', 'buffalo'], ['mole_rat', 'giraffe'], ['mole_rat', 'elephant'], ['warthog'],
    ],
  },
  {
    id: 'diverse_savanna', name: 'Diverse Savanna',
    ratioR1: 50, ratioR2: 30, ratioR3: 20,
    color: '#98c428',
    packs: [
      ['giraffe', 'mole_rat', 'gazelle'], ['giraffe', 'mole_rat', 'impala'],
      ['giraffe', 'mole_rat', 'buffalo'], ['giraffe', 'warthog', 'impala'],
      ['giraffe', 'warthog', 'buffalo'], ['giraffe', 'mole_rat', 'zebra'],
    ],
  },
  {
    id: 'root_mosaic', name: 'Root Mosaic',
    ratioR1: 40, ratioR2: 20, ratioR3: 40,
    color: '#b88c20',
    packs: [
      ['giraffe', 'mole_rat', 'gazelle'], ['giraffe', 'mole_rat', 'impala'],
      ['giraffe', 'mole_rat', 'buffalo'], ['giraffe', 'warthog', 'impala'],
      ['giraffe', 'warthog', 'buffalo'], ['giraffe', 'mole_rat', 'zebra'],
    ],
  },
  {
    id: 'leaf_mosaic', name: 'Leaf Mosaic',
    ratioR1: 30, ratioR2: 50, ratioR3: 20,
    color: '#28a870',
    packs: [
      ['giraffe', 'mole_rat', 'gazelle'], ['giraffe', 'mole_rat', 'impala'],
      ['giraffe', 'mole_rat', 'buffalo'], ['giraffe', 'warthog', 'buffalo'],
    ],
  },
];

const BIOME_MAP = new Map<string, BiomeData>(BIOMES.map(b => [b.id, b]));

export function getBiome(id: string): BiomeData {
  const b = BIOME_MAP.get(id);
  if (!b) throw new Error(`Unknown biome: ${id}`);
  return b;
}
