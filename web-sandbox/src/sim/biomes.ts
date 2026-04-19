export interface BiomeData {
  id: string;
  name: string;
  ratioR1: number; // 0-100
  ratioR2: number;
  ratioR3: number;
  color: string;  // CSS hex color
  packs: string[][];
}

// Colors derived from linear interpolation of base pigments:
//   grass=(165,225,50)  leaf=(20,80,15)  root=(155,85,20)
//   color = (r1/100)*grass + (r2/100)*leaf + (r3/100)*root
export const BIOMES: BiomeData[] = [
  {
    id: 'grassland', name: 'Grassland',
    ratioR1: 100, ratioR2: 0, ratioR3: 0,
    color: '#A5E132',  // pure grass
    packs: [[], ['gazelle'], ['zebra'], ['buffalo']],
  },
  {
    id: 'open_savanna', name: 'Open Savanna',
    ratioR1: 67, ratioR2: 33, ratioR3: 0,
    color: '#75B126',  // grass-dominant mix
    packs: [
      ['giraffe'], ['giraffe', 'gazelle'], ['giraffe', 'zebra'],
      ['giraffe', 'buffalo'], ['giraffe', 'warthog'], ['giraffe', 'impala'], ['impala'],
    ],
  },
  {
    id: 'woodland', name: 'Woodland',
    ratioR1: 33, ratioR2: 67, ratioR3: 0,
    color: '#44801B',  // leaf-dominant, mid-dark green
    packs: [['giraffe'], ['giraffe', 'buffalo'], ['giraffe', 'impala']],
  },
  {
    id: 'dense_grove', name: 'Dense Grove',
    ratioR1: 20, ratioR2: 80, ratioR3: 0,
    color: '#316D16',  // heavy leaf, dark forest
    packs: [['giraffe'], ['elephant']],
  },
  {
    id: 'root_patch', name: 'Root Patch',
    ratioR1: 67, ratioR2: 0, ratioR3: 33,
    color: '#A2B328',  // grass+root, olive-yellow
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
    color: '#9E831E',  // root-dominant, warm brown
    packs: [
      ['mole_rat'], ['mole_rat', 'gazelle'], ['mole_rat', 'zebra'],
      ['mole_rat', 'buffalo'], ['mole_rat', 'giraffe'], ['mole_rat', 'elephant'], ['warthog'],
    ],
  },
  {
    id: 'diverse_savanna', name: 'Diverse Savanna',
    ratioR1: 50, ratioR2: 30, ratioR3: 20,
    color: '#789A22',  // balanced mix, yellow-green
    packs: [
      ['giraffe', 'mole_rat', 'gazelle'], ['giraffe', 'mole_rat', 'impala'],
      ['giraffe', 'mole_rat', 'buffalo'], ['giraffe', 'warthog', 'impala'],
      ['giraffe', 'warthog', 'buffalo'], ['giraffe', 'mole_rat', 'zebra'],
    ],
  },
  {
    id: 'root_mosaic', name: 'Root Mosaic',
    ratioR1: 40, ratioR2: 20, ratioR3: 40,
    color: '#848C1F',  // even root/grass mix, olive
    packs: [
      ['giraffe', 'mole_rat', 'gazelle'], ['giraffe', 'mole_rat', 'impala'],
      ['giraffe', 'mole_rat', 'buffalo'], ['giraffe', 'warthog', 'impala'],
      ['giraffe', 'warthog', 'buffalo'], ['giraffe', 'mole_rat', 'zebra'],
    ],
  },
  {
    id: 'leaf_mosaic', name: 'Leaf Mosaic',
    ratioR1: 30, ratioR2: 50, ratioR3: 20,
    color: '#5B7D1B',  // leaf-heavy with some root, dark olive
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
