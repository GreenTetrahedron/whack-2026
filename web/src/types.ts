// Shapes returned by the FastAPI backend (khatt/api/main.py, khatt/evaluate.py).

export type LatLng = { lat: number; lng: number };

export type Health = {
  ok: boolean;
  n_hexes: number;
  n_streets: number;
  t0_shape: number[];
  synthetic: boolean;
  load_s: number;
  error?: string | null;
};

export type Category = "daily" | "comp" | "eat";

export type BaselineStreet = {
  street_id: number;
  name: string;
  lat: number;
  lng: number;
  R0: Record<Category, number>;
  R0_total: number;
  n_shops: Record<Category, number>;
  n_unbranded: number;
};

export type Feature<P> = {
  type: "Feature";
  properties: P;
  geometry: { type: string; coordinates: unknown };
};
export type FeatureCollection<P> = { type: "FeatureCollection"; features: Feature<P>[] };

export type Baseline = {
  streets: BaselineStreet[];
  existing_network: FeatureCollection<{ line: string }>;
  existing_stations: { stop_id: string; line: string; name: string; lat: number; lng: number; catchment_pop10: number }[];
  microbus_routes: FeatureCollection<{ route_id: string; agency: string; headway_min: number }>;
  credits: string;
};

export type Risk = "green" | "amber" | "red";

export type StreetResult = {
  street_id: number;
  name: string;
  lat: number;
  lng: number;
  R0_egp: number;
  dR_egp: number;
  dR_pct: number;
  by_category: Record<Category, number>;
  by_category_egp: Record<Category, number>;
  passby_egp: number;
  rent_pct: number;
  rent_pct_opening: number;
  revenue_pct: number;
  risk: Risk;
  risk_opening: Risk;
  local_loser: boolean;
  unbranded_at_risk: number;
  flows?: { from: [number, number]; egp: number }[]; // this street's own largest shifts (missing in old scenario files)
};

export type StationResult = {
  line_id: string;
  idx: number;
  lat: number;
  lng: number;
  h3: string;
  riders_daily: number;
  catchment_pop10: number;
  usefulness: number;
  vendor_pitches: number;
  snap_m: number;
  zone_radius_m: number; // area of effect: 0 = no zone
  zone_users: number; // residents × chance a typical trip boards here
  zone_pop: number; // residents inside the circle
  zone_mean_p: number; // mean chance inside the circle
};

export type EvalResult = {
  elapsed_ms: number;
  warnings: string[];
  lines: { line_id: string; name: string; n_stations: number; length_km: number; headway_min: number; speed_kmh: number }[];
  totals: {
    dR_egp: number;
    abs_dR_egp: number;
    R0_egp: number;
    passby_egp: number;
    total_revenue_change: number;
    riders_daily: number;
    streets_gaining: number;
    streets_losing: number;
    local_losers: number;
    red_streets: number;
    amber_streets: number;
    red_streets_opening: number;
    unbranded_at_risk: number;
    max_rent_pct: number;
    vendor_pitches: number;
    vendor_pitches_lost: number;
    microbus: { competing: number; feeder: number; unaffected: number };
    zone_users: number;
    zone_threshold: number;
  };
  streets: StreetResult[];
  stations: StationResult[];
  hexes: { h3: string; d_spend_reach: number; rent_pct: number; rent_pct_opening: number }[];
  footfall_cells: { h3: string; footfall_daily: number }[];
  flows: { from: [number, number]; to: [number, number]; street_id: number; egp: number }[];
  microbus_routes: { route_id: string; status: "competing" | "feeder" | "unaffected"; share_at_risk: number }[];
  synergy: { lines: string[]; abs_dR_egp: number }[];
  attribution: Record<string, Record<string, number>>;
  timings_ms: Record<string, number>;
};

export type ParamRange = { key: string; low: number; central: number; high: number; value: number; source: string };

// A line the user is drawing (front-end state only; the API is stateless).
export type DraftLine = {
  id: string;
  name: string;
  color: string;
  stations: LatLng[];
  headway_min: number;
  speed_kmh: number;
  visible: boolean;
};
