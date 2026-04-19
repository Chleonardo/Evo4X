export type Trend = 'up' | 'down' | 'flat';

export interface SimConfig {
  totalHexes: number;
  numRegions: number;
  minRegionSize: number;
  richnessPerHex: number;
  startPopDensity: number;
  extinctionEps: number;
  bEnableEvents: boolean;
  trendHistoryLen: number;
  trendWindowShort: number;
  trendThreshold: number;
  worldSeed: string;
}

export const DEFAULT_CONFIG: SimConfig = {
  totalHexes: 500,
  numRegions: 20,
  minRegionSize: 7,
  richnessPerHex: 10,
  startPopDensity: 2,
  extinctionEps: 1.0,
  bEnableEvents: false,
  trendHistoryLen: 50,
  trendWindowShort: 3,
  trendThreshold: 0.05,
  worldSeed: '42',
};

export interface HexTile {
  index: number;
  q: number;
  r: number;
  wx: number;
  wy: number;
  isLand: boolean;
  regionId: number;
}

export interface HexGrid {
  tiles: HexTile[];
  coordMap: Map<string, number>;
  landIndices: number[];
  hexSize: number;
}

export interface ActiveEffect {
  effectType: string;
  multR1: number;
  multR2: number;
  multR3: number;
  ttl: number;
}

export interface SpeciesInRegion {
  population: number;
  popPreBirth: number;
  foodAllocated: number;
  starvingTotal: number;
  starvingNewborns: number;
  outgoingMigrants: number;
  incomingMigrants: number;
  trend: Trend;
}

export interface MigrationEdge {
  fromRegion: number;
  toRegion: number;
  speciesId: string;
  count: number;
}

export interface Region {
  id: number;
  hexIndices: number[];
  hexCount: number;
  biomeId: string;
  neighbors: number[];
  baseR1: number;
  baseR2: number;
  baseR3: number;
  effR1: number;
  effR2: number;
  effR3: number;
  populations: Map<string, number>;
  tickBreakdown: Map<string, SpeciesInRegion>;
  activeEffects: ActiveEffect[];
  popHistory: Map<string, number[]>;
  dominantSpecies: string;
  totalBiomass: number;
  biomassTrend: Trend;
  centroidX: number;
  centroidY: number;
}

export interface GlobalMetrics {
  totalWorldBiomass: number;
  aliveSpeciesCount: number;
  fastestGrowing: string;
  fastestDeclining: string;
  speciesTotalPop: Map<string, number>;
}

export interface WorldState {
  currentTick: number;
  config: SimConfig;
  grid: HexGrid;
  regions: Region[];
  pendingMigrants: Map<number, Map<string, number>>;
  migrationEdgesThisTick: MigrationEdge[];
  migrationSmoothed: Map<string, MigrationEdge>; // key: "fromRegion->toRegion:speciesId"
  metrics: GlobalMetrics;
  globalPopHistory: Map<string, number[]>;
}
