# CLAUDE.md — Khatt (Cairo Metro What-If)

You are implementing **Khatt** ("line" in Arabic) for a 14-hour hackathon (Marshall Wace track, theme "New Worlds").
A user draws one or more hypothetical metro lines anywhere in Greater Cairo, clicks **Construct line**, and within ~2 seconds sees how local businesses are affected: spending shifts between shopping streets, footfall around stations, rent pressure / small-shop displacement, and impacts on microbuses and street vendors. Multiple lines are evaluated jointly (cumulative + synergy).

Read this whole file before writing code. When something here conflicts with your instincts, follow this file. When something is ambiguous, pick the simplest option that keeps `/evaluate` under 2 s and note the choice in `DECISIONS.md`.

---

## 1. Hard rules

1. **Build in rungs (section 12).** Every rung must leave a working demo. Never start rung N+1 while rung N is broken.
2. **`POST /evaluate` must return in < 2 s** for up to 4 lines / 40 stations total, and the response must be < 2 MB.
3. **Never fetch data at API runtime.** All heavy data is precomputed by `pipeline/` scripts into `data/processed/` and loaded into memory at API start-up.
4. **Every model parameter lives in `config/params.yaml`** with `low`, `central`, `high` values and a `source` note ("assumption" if invented). No magic numbers in code.
5. **Units:** time in **minutes** (float32), distance in **metres**, money in **EGP at 2021 prices**. Always return % changes alongside EGP.
6. **Baseline = Cairo today, without Metro Line 4.** Line 4 is not open (due 2028) and must not be in the baseline network.
7. **Coordinates:** store WGS84 (EPSG:4326). Do all metric work (buffers, distances, DBSCAN) in **EPSG:32636 (UTM 36N)**. Axis order: GeoJSON / deck.gl = `[lng, lat]`; `h3` functions take `(lat, lng)`. Name variables `lat`, `lng` explicitly — never `x`, `y` for geographic coordinates.
8. **Unreachable travel time = 999.0**, never NaN or inf (NaN breaks `min`).
9. **Licences:** Transport for Cairo (TfC) data is CC BY-NC 4.0 — show "Transit data: Transport for Cairo (CC BY-NC 4.0)" in the web footer. WorldPop CC BY 4.0. Overture Places CDLA-Permissive-2.0. OSM ODbL.
10. Raw data is never committed. `data/raw/` and `data/interim/` are git-ignored.

---

## 2. Stack

- **Python 3.11**: numpy, numba, pandas, geopandas, shapely, pyproj, h3 (**v4 API**: `latlng_to_cell`, `cell_to_latlng`, `grid_disk`), osmnx, networkx, scipy, scikit-learn (DBSCAN), rasterio, duckdb, overturemaps, r5py (needs a Java JDK), fastapi, uvicorn, pydantic v2, pyyaml, pytest, ruff.
- **Web**: Vite + React + TypeScript, MapLibre GL, deck.gl (`H3HexagonLayer`, `PathLayer`, `ScatterplotLayer`, `ArcLayer`, `TextLayer`), Recharts for charts.
- Pin library versions in `pyproject.toml` / `package.json`.

---

## 3. Repository layout

```
khatt/
├── CLAUDE.md
├── DECISIONS.md                 # running log of assumptions/choices you made
├── Makefile
├── pyproject.toml
├── config/
│   └── params.yaml              # every model parameter (low/central/high/source)
├── data/
│   ├── raw/                     # downloads (git-ignored)
│   ├── interim/                 # cleaned intermediates (git-ignored)
│   └── processed/               # files the API loads (small ones may be committed)
├── pipeline/                    # run once, in order; each script is idempotent
│   ├── 00_download.py
│   ├── 01_grid.py
│   ├── 02_population.py
│   ├── 03_spending.py
│   ├── 04_shops.py
│   ├── 05_streets.py
│   ├── 06_walk_graph.py
│   ├── 07_gtfs_clean.py
│   ├── 08_baseline_matrix.py
│   ├── 09_existing_stations.py
│   └── 10_microbus_routes.py
├── khatt/                       # importable package used by API + tests
│   ├── config.py                # loads params.yaml; Params dataclass; sample() for Monte Carlo
│   ├── data.py                  # loads data/processed into a single `World` object (cached)
│   ├── geo.py                   # projections, haversine, snapping helpers
│   ├── engine/
│   │   ├── lines.py             # Line/Station dataclasses, snapping, ride-time matrix
│   │   ├── multiline.py         # joint station graph + Floyd–Warshall
│   │   ├── overlay.py           # min-plus overlay (numba)
│   │   └── catchment.py         # walking catchments on the OSM graph
│   ├── models/
│   │   ├── spending.py          # Huff model (model 1)
│   │   ├── footfall.py          # station riders + pass-by spending (model 2)
│   │   ├── rents.py             # rent pressure + margin squeeze (model 3)
│   │   └── informal.py          # microbus routes + vendors (model 4)
│   ├── evaluate.py              # orchestrates engine + 4 models → result dict
│   ├── attribution.py           # synergy + Shapley split across lines
│   ├── montecarlo.py            # offline uncertainty runs for hero scenarios
│   └── api/
│       ├── main.py              # FastAPI app
│       └── schemas.py           # pydantic request/response models
├── scenarios/                   # precomputed hero scenario JSONs (demo fallback)
├── scripts/
│   ├── precompute_scenarios.py
│   └── backtest_line3.py
├── tests/
└── web/                         # Vite React app
```

---

## 4. Raw data sources (`pipeline/00_download.py`)

Download into `data/raw/`. If a download needs a login or fails, print a clear message telling the human what to fetch manually and where to put it; do not silently continue.

| Key | Source | Notes |
|---|---|---|
| `gtfs_bus` | Transport for Cairo Greater Cairo GTFS, Mobility Database feed `mdb-1825` (producer URL `https://data.transportforcairo.com/documents/41/download`) | ~930 bus/paratransit routes, service range Jan 2021–Jun 2022, frequency-based |
| `gtfs_metro` | Transport for Cairo metro GTFS 2022 (`https://data.transportforcairo.com/documents/42`) | 3 lines as of Jul 2022 |
| `osm` | Geofabrik Egypt `.osm.pbf` (for r5py) + OSMnx download for the walk graph | |
| `worldpop` | `egy_pop_2026_CN_100m_R2025A_v1.tif` from `data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2026/EGY/v1/100m/constrained/` | 100 m constrained population, 2026 |
| `overture_places` | `overturemaps` Python package / DuckDB over Overture GeoParquet, bbox below | Use the latest release; keep `confidence`, `categories`, `brand`, `operating_status` (if present), `names` |
| `overture_buildings` | same, type `building` | footprint area only |
| `osm_shops` | OSMnx `features_from_bbox` with `shop`, `amenity in {restaurant, cafe, fast_food, marketplace}` | |
| `spending` | CAPMAS HIECS 2021 published tables (`censusinfo.capmas.gov.eg`, catalog 747) | If tables can't be fetched automatically, read `data/raw/spending_manual.csv` (human fills it) |
| `wealth` (optional) | Meta Relative Wealth Index for Egypt, else VIIRS night lights | If neither is available, wealth factor = 1.0 everywhere and log it |

**Greater Cairo bounding box** (lng_min, lat_min, lng_max, lat_max): `(30.85, 29.80, 31.85, 30.35)`. Put it in `params.yaml` under `area.bbox` so it can be adjusted.

---

## 5. Processed data contracts (`data/processed/`)

These file names, columns and dtypes are the contract between pipeline, API and web. Do not change them without updating this section.

### `hex_index.json`
Ordered list of H3 resolution-8 cell IDs (strings). Row/column `k` of every matrix refers to `hex_index[k]`. Only cells with population > 0 **or** containing a shopping street **or** a transit stop. Expect ~3,000 cells.

### `hexes_r8.parquet`
| column | dtype | meaning |
|---|---|---|
| `h3` | str | cell id |
| `idx` | int32 | position in `hex_index.json` |
| `lat`, `lng` | float64 | cell centroid |
| `governorate` | str | Cairo / Giza / Qalyubia |
| `pop` | float32 | residents (WorldPop sum, preserves raster total) |
| `wealth` | float32 | relative wealth factor, population-weighted mean = 1.0 |
| `spend_daily` | float32 | EGP/yr, 2021 prices, daily needs (COICOP 01) |
| `spend_comp` | float32 | EGP/yr, comparison goods (COICOP 03, 05, electronics part of 09) |
| `spend_eat` | float32 | EGP/yr, eating out (COICOP 11) |

### `streets.parquet` (+ `streets.geojson` for the web)
| column | dtype | meaning |
|---|---|---|
| `street_id` | int32 | stable id |
| `name` | str | most common street name among member shops, else "Cluster {id}" |
| `lat`, `lng` | float64 | cluster centroid |
| `h3_r8`, `h3_r9` | str | cells containing the centroid |
| `hex_idx` | int32 | index into `hex_index.json` |
| `n_daily`, `n_comp`, `n_eat` | int32 | shops per category |
| `n_unbranded` | int32 | shops with no brand |
| `footprint_m2` | float32 | summed building footprint of member shops (proxy floorspace) |
| `A_daily`, `A_comp`, `A_eat` | float32 | attractiveness (default = shop counts; see params) |
| `geometry` | point | in GeoJSON only |

### `T0.npy`
`float32[N, N]`, minutes, door-to-door walk + public transport, weekday 08:00, from hex `i` to hex `j`. Diagonal = `access.intracell_min`. Unreachable = 999.0.

### `walk_graph.pkl`
Projected (EPSG:32636) OSMnx graph, largest strongly connected component, edge attribute `walk_min`. Plus `walk_nodes_r9.parquet` mapping `node_id → h3_r9`.

### `existing_stations.parquet`
`stop_id, line, name, lat, lng, h3_r8, hex_idx, catchment_pop10` (residents within a 10-min walk, each hex assigned to its nearest station only once).

### `microbus_routes.parquet`
`route_id, agency, headway_min, stop_hex_idx (list[int32]), geometry (LineString)`.

### `r9_near.parquet` (built lazily is fine)
Resolution-9 cells with `pop`, `parent_r8_idx`, used for station catchments and fine-grained footfall/rent output.

---

## 6. Pipeline details

### `01_grid.py`
Polyfill the bbox at res 8 (`h3.geo_to_cells` or `polygon_to_cells`), keep cells per section 5 rules after steps 02/05/07 (re-filter at the end of 07).

### `02_population.py`
Read WorldPop raster with rasterio, mask nodata, convert each populated pixel centre to its res-8 (and res-9) cell, **sum** counts. Assert `abs(sum(hex pop) - raster sum in bbox) / raster sum < 0.01`.

### `03_spending.py`
1. Load per-capita annual spending per governorate and COICOP division (urban). Confirm units (annual vs monthly; per household vs per person; divide household figures by mean household size). Log the conversion in `DECISIONS.md`.
2. Categories: `daily` = 01; `comp` = 03 + 05 + (09 × `params.spending.share_09_electronics`); `eat` = 11. Exclude rent, utilities, health, education, transport.
3. Allocate dasymetrically: `spend_c[h] = pop[h] × per_capita_c[gov] × wealth[h]`, then rescale within each governorate so totals equal `pop_gov × per_capita_c[gov]`.

### `04_shops.py`
1. Load Overture places in bbox. Filter `confidence >= params.shops.min_confidence` and drop closed (`operating_status`) if the field exists.
2. Load OSM shops/amenities.
3. Map categories → `daily | comp | eat | other` with an explicit crosswalk table `config/category_crosswalk.csv` (columns `source, raw_category, khatt_category`). Unmapped → `other` (kept for counts, excluded from A).
4. Deduplicate across sources: same khatt category, within 40 m (projected), and name similarity ≥ 0.8 after normalisation.
5. **Arabic name normalisation** before matching: strip tashkeel (U+064B–U+0652) and tatweel (U+0640); map أ إ آ → ا; ة → ه; ى → ي; lowercase Latin; collapse whitespace.
6. `brand` missing → `unbranded=True` (do not call it "independent" in the UI).

### `05_streets.py`
DBSCAN on projected shop coordinates (`eps = params.streets.eps_m`, default 150; `min_samples = params.streets.min_shops`, default 15). Noise points are dropped. Compute section-5 columns. Join building footprints within 20 m of member shops for `footprint_m2`. Write `streets.parquet` and `streets.geojson`.

### `06_walk_graph.py`
`osmnx.graph_from_bbox(..., network_type="walk")`, keep largest strongly connected component, project to EPSG:32636, `walk_min = length / (params.walk.speed_kmh * 1000/60)`. Check: the Nile islands (Zamalek, Roda) are in the component; log a warning otherwise.

### `07_gtfs_clean.py`
1. Run MobilityData's `gtfs-validator` on each feed (if Java available); save reports to `data/interim/validation/`. Fix or drop routes with errors.
2. Checks: `stop_times` non-decreasing within a trip (times > 24:00:00 are valid); stops inside bbox (if a stop is outside, try swapping lat/lng); `frequencies.txt` with `exact_times = 0` treated as headway-based.
3. Remove metro routes (`route_type` 1) from the bus feed if they duplicate the metro feed. Prefix all IDs with the feed key (`bus:`, `metro:`, `manual:`).
4. Add a `manual` feed (`config/manual_lines.yaml` → GTFS) for lines missing from the 2022 metro feed: **Line 3 west branch (Kit Kat → Rod El Farag, opened Oct 2022) and the final section to Cairo University (opened May 2024)**; optionally the LRT and East Nile monorail (Mar 2026). Station coordinates are entered by a human in the YAML; if empty, skip with a warning.
5. Calendar: every feed must have service on `params.routing.departure_date` (default `2022-03-15`, a Tuesday inside the bus feed's range). For manual feeds, generate a calendar covering that date. Do **not** change real feeds' dates.

### `08_baseline_matrix.py` (longest job — start it as early as possible)
- r5py `TransportNetwork(osm_pbf, [gtfs files])`, `TravelTimeMatrixComputer` (or current r5py equivalent) with origins = destinations = res-8 centroids, departure `params.routing.departure_date` 08:00, `departure_time_window = 60 min`, transport modes `[TRANSIT, WALK]`, median travel time.
- Set Java heap via r5py's `--max-memory` / environment before import. Log runtime.
- Output `T0.npy` (float32, minutes, 999 for missing) with the `hex_index.json` order.
- **Fallback (if r5py fails or takes > 90 min):** build a custom graph with scipy: nodes = res-8 cells + stops; walk edges between neighbouring cells (`grid_disk` k=1, time = centroid distance × `walk.circuity` / speed); stop–cell access edges; transit edges between consecutive stops with ride time from GTFS and boarding wait = headway/2 × `routing.irregularity_factor[mode]`. Run `scipy.sparse.csgraph.dijkstra` from every cell. Same output contract.

### `09_existing_stations.py`
For every existing metro/LRT/monorail station: walking catchment (10 min) on the walk graph, assign each res-9 cell to its nearest station only (no double counting), sum population → `catchment_pop10`.

### `10_microbus_routes.py`
From the bus feed, routes whose agency is paratransit/microbus. Store ordered stop hexes and headway.

---

## 7. Engine (`khatt/engine/`)

### Line input
```python
@dataclass
class LineSpec:
    line_id: str
    name: str
    stations: list[tuple[float, float]]   # (lat, lng), in order, >= 2
    headway_min: float                    # default params.line.headway_min
    speed_kmh: float                      # average running speed, default params.line.speed_kmh
    dwell_s: float                        # per intermediate stop, default params.line.dwell_s
```

### Ride times (`lines.py`)
Cumulative haversine distance along the station sequence. `ride[i, j] = dist(i, j) / speed + dwell × (number of intermediate stops)`, symmetric, minutes. Snap each station to its res-8 cell (`hex_idx`) and to the nearest walk-graph node (for catchments); if the snap distance is > 200 m, flag it in the response (`warnings`).

### Joint station graph (`multiline.py`)
Nodes = all stations of all drawn lines (S total). Edges:
- along each line: `ride[i, j]` for consecutive stations;
- transfers between any two stations `a ≠ b` (different lines or same location): `T0[hex_a, hex_b] + wait_b + board_penalty` where `wait_b = headway_b / 2`.

Run Floyd–Warshall (numpy, S ≤ 150) → `Rstar[i, j]` = best time from *on board at i* to *alighting at j*, including any interchanges. `Rstar[i, i] = 0`.

### Overlay (`overlay.py`) — the core formula
For origin `o` and destination `d`:

```
T1[o, d] = min( T0[o, d],
                min_{i, j} ( T0[o, s_i] + wait_i + board_penalty + Rstar[i, j] + T0[s_j, d] ) )
```

`T0[o, s_i]` uses `hex_idx` of station `i`; when `o == hex_idx(s_i)` this is the intra-cell access time (diagonal).

Implement with **numba** (`@njit(parallel=True, fastmath=False)`), parallel over origins, two passes to stay O(N·S² + N·S·D):

```python
@njit(parallel=True)
def overlay(T0_od, T0_os, T0_sd, wait_board, Rstar):
    # T0_od: (N, D) baseline origin->destination
    # T0_os: (N, S) baseline origin->station cell
    # T0_sd: (S, D) baseline station cell->destination
    # wait_board: (S,) wait + board penalty
    # Rstar: (S, S)
    N, D = T0_od.shape
    S = Rstar.shape[0]
    T1 = T0_od.copy()
    for o in prange(N):
        via = np.full(S, 1e9, np.float32)       # best arrival at alighting station j
        for i in range(S):
            a = T0_os[o, i] + wait_board[i]
            for j in range(S):
                v = a + Rstar[i, j]
                if v < via[j]:
                    via[j] = v
        for d in range(D):
            best = T1[o, d]
            for j in range(S):
                v = via[j] + T0_sd[j, d]
                if v < best:
                    best = v
            T1[o, d] = best
    return T1
```

Destinations `D` = shopping-street cells (for models 1–2) plus res-8 cells within `rents.radius_m` of any new station (for model 3). Never compute the full N×N T1 per request.

### Catchments (`catchment.py`)
Single-source Dijkstra on the in-memory walk graph from each new station's snapped node with cutoff `footfall.catchment_min` (10) and `footfall.passby_radius_m` (as minutes). Return reached res-9 cells with walk times. Cache by node id.

### Engine invariants (must have tests)
- `T1 <= T0 + 1e-4` everywhere (adding a line can never make a trip slower).
- No lines → `T1 == T0`.
- Result independent of the order lines were drawn.
- Overlay matches brute-force Dijkstra on a toy graph (≤ 30 nodes) to within 1e-4.

---

## 8. Models (`khatt/models/`)

All models take `(world, params, T0_sub, T1_sub, lines, catchments)` and return plain dicts / numpy arrays. Run each model for baseline and scenario, then diff.

### Model 1 — spending shifts (`spending.py`)
For category `c ∈ {daily, comp, eat}`:

```
logit[o, d] = alpha * ln(A[d, c]) - beta_c * T[o, d]          (exclude d where T > spending.max_trip_min)
P[o, d]     = softmax_d(logit[o, :])                            (subtract row max for stability)
R[d, c]     = sum_o S[o, c] * P[o, d]
```

- `S[o, c]` from `hexes_r8.spend_*`. `A` from `streets.A_*` (streets with A = 0 in a category are excluded for that category).
- Output per street: `R0`, `R1`, `dR = R1 - R0` (EGP/yr) and `dR_pct`, per category and total.
- `induced = params.spending.induced_pct` (default 0): if > 0, scale `S` for origins whose logsum accessibility improved by `induced × relative logsum gain`. Default must keep `sum(dR) ≈ 0`.
- **Two-way road flag:** street with `dR < 0` and within `spending.local_radius_m` (1500) of a new station → `"local_loser": true`.
- Also return the top 50 origin→street shifts (largest `|S × ΔP|`) for the web `ArcLayer`.

### Model 2 — footfall (`footfall.py`)
1. **Riders per resident** (calibration, computed once at start-up): `r = params.footfall.metro_daily_riders (4.5e6) / sum(existing_stations.catchment_pop10)`.
2. **New station riders:** `B_s = r × catchment_pop10(s) × usefulness_s`, where `usefulness_s` = share of spending-weighted trips from the station's catchment cells whose best path uses the new line (`T1 < T0 - 0.5`), divided by `footfall.reference_use_share`, clipped to `[0.2, 2.0]`. Daily exits ≈ boardings.
3. **Pass-by spending** for each shopping street within `passby_radius_m` (400 m walk) of station `s`: `exits_s × exp(-walk_m / passby_decay_m) × buy_rate × basket_egp × 365` → EGP/yr. Sum over stations.
4. Do not add pass-by spend into model 1's `R`; report it separately and sum only in `total_revenue_change`.
5. Output: per station `{riders_daily, catchment_pop10}`, per street `passby_egp`, plus res-9 footfall cells for a heatmap.

### Model 3 — rents and displacement (`rents.py`)
1. **Market potential** of destination cell `h`: `M[h] = sum_c sum_o S[o, c] × exp(-beta_c × T[o, h])`, for baseline and scenario, only for cells within `rents.radius_m` of new stations.
2. `rent_pct[h] = elasticity × (M1 - M0) / M0` (plus `footfall_term × passby_share`).
3. **Margin squeeze per street:** `revenue_pct = (dR_total + passby_egp) / R0_total`. `profit_signal = gross_margin × revenue_pct − rent_cost_share × rent_pct`. Risk: `green` if `profit_signal ≥ 0`, `amber` if `≥ −rents.red_threshold`, else `red`.
4. Output per street: `rent_pct`, `revenue_pct`, `risk`, `unbranded_at_risk = n_unbranded if risk == red else 0`. Label rents as an **index** in the UI, never as EGP.
5. Report two horizons: `opening` (rent_pct × `rents.short_run_share`) and `settled` (full).

### Model 4 — microbuses and vendors (`informal.py`)
1. For each microbus route, take stop-hex pairs `(a, b)` with `a` before `b` and at most `informal.max_pair_gap` stops apart. `share_at_risk = mean(T1[a, b] < T0[a, b] − informal.switch_margin_min)`.
2. Classify: `competing` if `share_at_risk ≥ informal.compete_threshold`; else `feeder` if any route stop is within `informal.feeder_radius_m` of a new station; else `unaffected`.
3. Vendors: `vendor_pitches_s = exits_s / 1000 × informal.vendors_per_1000_exits` at each new station. Report `vendor_pitches_lost` at the busiest stops of competing routes proportional to `share_at_risk` (simple, labelled "estimate").
4. Output: counts per class, per-route `{route_id, status, share_at_risk}` with geometry ids for the map, vendor totals.

---

## 9. Multi-line attribution (`attribution.py`)
- **Joint evaluation** is always the source of truth.
- **Synergy** for each pair of lines: `effect(A ∪ B) − effect(A) − effect(B)` on total `|dR|` and per street.
- **Shapley split** (≤ 4 lines → ≤ 16 subsets; skip with a warning above 4): for each street, each line's Shapley value of `dR_total`. Assert parts sum to the joint value (tolerance 1e-3 × |joint|).
- Expensive subsets run only when the request sets `"attribution": true`.

---

## 10. API (`khatt/api/`)

Load `World` once at start-up (`lifespan`). Enable CORS for the Vite dev server. Convert numpy types to Python before returning.

| Method / path | Request | Response |
|---|---|---|
| `GET /health` | — | `{ok, n_hexes, n_streets, t0_shape}` |
| `GET /baseline` | — | streets (id, name, lat, lng, R0 per category), existing network geometry (GeoJSON), microbus route geometries (simplified) |
| `POST /evaluate` | `{lines: LineSpec[], params?: {...overrides}, attribution?: bool}` | see below |
| `GET /scenarios` | — | list of precomputed hero scenarios |
| `GET /scenarios/{id}` | — | stored `/evaluate` response (+ Monte Carlo ranges) |
| `GET /params` | — | current central values and low/high for UI sliders |

The frontend owns line state; the API is stateless. `/evaluate` response shape:

```json
{
  "elapsed_ms": 830,
  "warnings": ["Station 3 of 'Line A' is 340 m from the walking network"],
  "totals": {"dR_egp": 0.0, "passby_egp": 1.2e8, "riders_daily": 310000,
             "streets_gaining": 41, "streets_losing": 37, "red_streets": 6,
             "unbranded_at_risk": 212, "microbus": {"competing": 14, "feeder": 22, "unaffected": 894}},
  "streets": [{"street_id": 12, "name": "...", "dR_egp": -3.4e6, "dR_pct": -4.1,
               "by_category": {"daily": -0.2, "comp": -5.9, "eat": -2.0},
               "passby_egp": 0, "rent_pct": 1.8, "revenue_pct": -4.1,
               "risk": "red", "local_loser": true, "unbranded_at_risk": 9}],
  "stations": [{"line_id": "A", "idx": 0, "lat": 30.0, "lng": 31.2, "riders_daily": 42000, "vendor_pitches": 84}],
  "hexes": [{"h3": "8834...", "d_spend_reach": 0.03, "rent_pct": 2.1}],
  "flows": [{"from": [31.21, 30.01], "to": [31.24, 30.05], "egp": 2.1e6}],
  "microbus_routes": [{"route_id": "bus:123", "status": "competing", "share_at_risk": 0.46}],
  "synergy": [{"lines": ["A", "B"], "abs_dR_egp": 4.0e6}],
  "attribution": {"12": {"A": -2.1e6, "B": -1.3e6}}
}
```

Only include streets/hexes whose values changed (|dR_pct| > 0.1 or risk ≠ green) to keep the payload small.

---

## 11. Web app (`web/`)

- **Map** (MapLibre basemap + deck.gl overlay) centred on Cairo (lat 30.04, lng 31.24, zoom 10.5). Existing network in grey.
- **Draw mode:** click to add stations in order; drag to move; right-click to delete; show the polyline live. Side form: line name, headway (min), speed (km/h). Buttons: **Construct line**, **Clear**, **Undo**. Multiple lines in a list with colour, visibility toggle and delete.
- **Construct line** → `POST /evaluate` with *all* current lines. Show a spinner; display `elapsed_ms`.
- **Layers:** streets as circles sized by `R0`, coloured diverging by `dR_pct` (red loss / blue gain, neutral grey); `ArcLayer` for top flows; `H3HexagonLayer` for rent pressure (toggle); station markers sized by riders; competing microbus routes in orange, feeders in green.
- **Results panel tabs:** Spending | Footfall | Rents | Microbuses & vendors | Lines & synergy. Each tab: 3 headline numbers + a sortable table (winners/losers). Every number shows % and EGP; rents labelled "index".
- **Before/after toggle** and an **assumptions drawer** with sliders for `beta` (per category), `rents.elasticity`, `spending.induced_pct` → re-run `/evaluate` with overrides.
- **Scenario picker** loads `/scenarios/{id}` (works even if the engine is down).
- Footer: data credits and "Model estimates, 2021 EGP prices".

---

## 12. Build order (rungs and gates)

| Hour | Rung / gate | Done when |
|---|---|---|
| 0–1 | Setup | repo, Makefile, params.yaml skeleton, downloads running, mock `/evaluate` JSON for the web team |
| 1–4 | Data | `hexes_r8.parquet`, `streets.parquet`, walk graph, cleaned GTFS; `08_baseline_matrix.py` running |
| **4** | **Gate 1** | `T0.npy` exists — else switch to the scipy fallback immediately |
| 4–8 | **Rung 1** | engine + model 1 + real `/evaluate` + map shows street winners/losers |
| **8** | **Gate 2** | draw a line → coloured streets in < 2 s, end to end |
| 8–9 | Rung 2 | model 2 (riders, pass-by spend) |
| 9–10 | Rung 3 | model 4 (microbuses, vendors) |
| 10–10.5 | Rung 4 | model 3 (rents, margin squeeze) |
| 10.5–12 | Rung 5 | multi-line synergy + Shapley; `precompute_scenarios.py` with Monte Carlo ranges; Line 3 backtest; polish |
| **12** | **Gate 3** | feature freeze — fixes only |

---

## 13. Calibration, validation, uncertainty

- **Metro total:** `footfall.metro_daily_riders = 4.5e6` (Ahram Online, Dec 2025) calibrates `r`.
- **β per category (Hyman's method):** bisection on `beta_c` until the baseline mean shopping trip time (spending-weighted) equals `spending.target_mean_trip_min[c]` (assumptions: daily 10, comp 30, eat 20). Store the fitted values in `data/processed/calibrated.json`; `params.yaml` low/high = ±50 % around the fitted value.
- **Backtest (`scripts/backtest_line3.py`):** build a T0 variant without the Line 3 west branch (rerun 08 with the manual feed excluded, or approximate by removing those stops in the fallback graph), then evaluate the branch as a drawn line. Save predicted riders and top winner/loser streets to `scenarios/backtest_line3.json`. Present as a plausibility test only.
- **Monte Carlo (`montecarlo.py`, offline only):** sample every parameter from `triangular(low, central, high)`, seed = 42, `N = 200` (500 if time). For hero scenarios report P10/P50/P90 for headline totals and, per street, the fraction of runs in which it is a loser (`p_loss`).

---

## 14. `config/params.yaml` (starting values — all assumptions unless stated)

```yaml
area:
  bbox: [30.85, 29.80, 31.85, 30.35]   # lng_min, lat_min, lng_max, lat_max
routing:
  departure_date: "2022-03-15"
  departure_time: "08:00"
  irregularity_factor: {metro: 1.0, bus: 1.3, microbus: 1.5}   # E[W] = h/2 * (1 + CV^2)
walk:
  speed_kmh: {low: 4.0, central: 4.5, high: 5.0, source: assumption}
  circuity: {low: 1.2, central: 1.3, high: 1.4, source: assumption}
access:
  intracell_min: {low: 3, central: 4, high: 6, source: "approx. walk within a res-8 cell"}
line:
  headway_min: {low: 3, central: 4, high: 6, source: assumption}
  speed_kmh: {low: 30, central: 35, high: 40, source: "assumption; tune vs existing lines"}
  dwell_s: {low: 20, central: 30, high: 45, source: assumption}
  board_penalty_min: {low: 2, central: 3, high: 5, source: assumption}
spending:
  alpha: {low: 0.8, central: 1.0, high: 1.2, source: assumption}
  target_mean_trip_min: {daily: 10, comp: 30, eat: 20}
  max_trip_min: 120
  induced_pct: {low: 0.0, central: 0.0, high: 0.05, source: assumption}
  share_09_electronics: {low: 0.2, central: 0.3, high: 0.4, source: assumption}
  local_radius_m: 1500
streets:
  eps_m: {low: 100, central: 150, high: 250}
  min_shops: 15
shops:
  min_confidence: 0.5
footfall:
  metro_daily_riders: {low: 4.0e6, central: 4.5e6, high: 5.0e6, source: "Ahram Online Dec 2025"}
  catchment_min: 10
  reference_use_share: {low: 0.2, central: 0.3, high: 0.4, source: assumption}
  passby_radius_m: 400
  passby_decay_m: {low: 150, central: 200, high: 300, source: assumption}
  buy_rate: {low: 0.02, central: 0.05, high: 0.10, source: assumption}
  basket_egp: {low: 15, central: 30, high: 50, source: "assumption, 2021 EGP"}
rents:
  radius_m: 1500
  elasticity: {low: 0.1, central: 0.3, high: 0.6, source: "placeholder; replace with published capitalisation estimates"}
  rent_cost_share: {low: 0.1, central: 0.2, high: 0.3, source: assumption}
  gross_margin: {low: 0.15, central: 0.25, high: 0.35, source: assumption}
  red_threshold: 0.02
  short_run_share: 0.3
informal:
  max_pair_gap: 15
  switch_margin_min: 5
  compete_threshold: 0.3
  feeder_radius_m: 400
  vendors_per_1000_exits: {low: 1, central: 2, high: 4, source: assumption}
```

---

## 15. Tests (`tests/`) — run with `make test`

- `test_overlay.py`: brute-force equivalence on a toy graph; `T1 <= T0`; no-lines identity; order independence.
- `test_multiline.py`: two crossing lines produce a transfer path shorter than either alone; Floyd–Warshall result symmetric for symmetric inputs.
- `test_spending.py`: probabilities sum to 1 per origin; `sum(dR) ≈ 0` when `induced_pct = 0`; a line far from all population changes nothing (|dR_pct| < 0.1 everywhere).
- `test_attribution.py`: Shapley parts sum to the joint effect.
- `test_data_contracts.py`: every processed file has the columns/dtypes in section 5; population total check; no NaN in `T0`.
- `test_performance.py`: `/evaluate` with 2 lines × 20 stations < 2 s on the dev laptop (mark as `slow`).

---

## 16. Makefile targets

```
make setup        # python env + web deps
make data         # pipeline 00–07, 09, 10
make baseline     # pipeline 08 (long) — run in background, logs to logs/baseline.log
make calibrate    # fit beta, write calibrated.json
make api          # uvicorn khatt.api.main:app --reload --port 8000
make web          # cd web && npm run dev
make scenarios    # precompute hero scenarios + Monte Carlo
make test
```

---

## 17. Hero scenarios to precompute (`scripts/precompute_scenarios.py`)

Store station coordinates in `config/scenarios.yaml` (humans will fill/adjust them):
1. **underserved** — a line through the densest area furthest from any existing station (find it programmatically: max population-weighted distance to nearest station; propose stations every ~1.2 km).
2. **orbital** — links outer ends of Lines 1, 2 and 3 (interchange synergy).
3. **two_way_road** — fast radial line into Downtown passing a local centre.
4. **synergy_pair** — two lines evaluated alone and together.
5. **faisal_haram** (optional reveal) — along Faisal/Haram streets.

---

## 18. Out of scope (do not build)

Car trips, road congestion, crowding/capacity, construction-period disruption, real-time data, user accounts, persistence of drawn lines on the server, mobile layout polish.

---

## 19. Working style

- Small commits per pipeline step / feature; commit message says what and why.
- After each pipeline script, print a 5-line summary (row counts, totals, runtime) and append notable choices to `DECISIONS.md`.
- If a data source is unavailable, implement the documented fallback and keep going; never block on it.
- Prefer clarity over cleverness except in `overlay.py`, where speed matters.
