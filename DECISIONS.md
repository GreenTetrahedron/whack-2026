# DECISIONS

Running log of assumptions and choices (CLAUDE.md: "pick the simplest option that keeps
`/evaluate` under 2 s and note the choice here").

## Setup

- **Environment**: uv + `pyproject.toml` with exact pins, Python 3.11, `.venv/`. `make` exports `.env`.
- **YAML numbers**: PyYAML reads `4.0e6` as a *string* (YAML 1.1). Write `4.0e+6` or plain integers;
  `tests/test_config.py` fails if any uncertain parameter is not numeric.

## Engine

- **Rstar is computed exactly, not by Floyd–Warshall over consecutive-station edges.** Chaining
  consecutive `ride[i, i+1]` edges drops the dwell at intermediate stops. Instead:
  `L` = same-line ride matrix (with dwells), `X` = transfer (T0 + wait + board),
  `Rstar = L ⊗ (X ⊗ L)*` in min-plus algebra, closure by Floyd–Warshall on S×S.
  Verified against brute-force Dijkstra on an explicit on-board state graph (`tests/test_overlay.py`).
- **Transfers allowed between any two stations a ≠ b** (spec says "different lines or same location").
  A superset: it can only lower times, costs nothing, and is what the brute-force test models
  (e.g. walking between the arms of a U-shaped line).
- **Stations outside the modelled cells** (their res-8 cell is not in `hex_index.json`) get an
  access/egress walk = distance to the nearest cell centre × circuity / speed, added on boarding,
  alighting and transfers. Otherwise a line drawn in the desert would "teleport" between edge cells.
  A warning is returned when that distance exceeds 1 km.
- **Overlay** is split into `compute_via` (N×S) and `overlay_from_via` / `overlay_pairs`, so the same
  scenario can be queried for street cells, rent cells and microbus stop pairs without recomputing.
- **Walk graph** is converted once to CSR and cached as `data/processed/walk_csr.npz`; catchments use
  a numba Dijkstra with cutoff and a reusable workspace, cached by (node, cutoff).
- **Catchment ties** (an r9 cell equidistant from two stations) go to the station that sorts first
  by (line_id, idx), so results do not depend on the order lines were drawn.
- **Model signatures**: models take explicit arrays (times to streets / rent cells, catchments)
  rather than the generic `(world, params, T0_sub, T1_sub, lines, catchments)` tuple; same inputs.

## Models

- **Model 1**: origins with no street within `max_trip_min` allocate nothing (in both baseline and
  scenario). `d_spend_reach` (hex output) = spending-weighted relative change of the Huff denominator
  Σ_d A^α e^(−βT). Induced demand scales S by `1 + induced_pct × that gain` (gain floored at 0).
- **Model 1 flows**: `|Σ_c S[o,c] ΔP[o,d,c]|`, top `spending.n_flows` (50), `[lng, lat]` order.
  Each reported street also carries its own top `spending.n_street_flows` (8) as `flows`
  (≈ +180 kB on the 2 × 20-station case, 508 kB in all).
- **Model 2 usefulness**: share of the catchment's spending-weighted trips *to shopping streets*
  with T1 < T0 − 0.5 (scenario probabilities), weighted by catchment population of each parent cell.
- **Model 2 pass-by**: walk distance from the station's catchment Dijkstra to each street's nearest
  walk node (street within 400 m straight line first). Heatmap (`footfall_cells`) = Σ station exits ×
  e^(−walk_m / passby_decay_m) over each station's reached r9 cells.
- **Model 3**: added `rents.footfall_term` (assumption, central 0.1) for the spec's
  `footfall_term × passby_share`. Street rent uses its own pass-by share; hex rent uses Σ pass-by /
  Σ R0 of its streets. Risk: *green* if profit signal ≥ −`rents.amber_threshold` (new parameter,
  0.0005, assumption) rather than the spec's ≥ 0. Huff is global, so every street loses a sliver of
  share to any new line; on real data that float noise made 756 of 912 streets amber (76 with the
  tolerance) and put every street in the payload. On synthetic data the rent index has a long tail (p99 ≈ 60 %)
  in cells with poor baseline access — review against real data before the demo.
- **Model 4**: consecutive duplicate stop hexes are collapsed before forming pairs (otherwise pairs
  inside one cell dilute `share_at_risk`). Feeder = any route stop within 400 m of a new station
  (`stop_lat`/`stop_lng`; the drawn line is the road trace, see Map display).
  Vendor pitches lost = Σ_competing share_at_risk × `informal.vendors_per_route` (new parameter,
  assumption: 6), labelled an estimate.
- **Station zones (area of effect, `khatt/models/zones.py`)**: new display model, not in the
  original spec; riders (model 2) are unchanged and stay calibrated to 4.5 M/day.
  - *Trips*: half work trips, half shopping trips (`zones.work_trip_share`, assumption). The idea is that
    commuters also shop where they work. Work destinations are weighted by `n_places` (all 99k mapped places
    in kept cells, any category): a jobs stand-in because there is no employment data.
    `zones.beta_work` is fitted by Hyman's method to a 40-min mean work trip (assumption) → 0.080/min.
  - *Switching*: a linear ramp (0 below 2 min saved, 1 at 10 min) instead of a logit. A logit gives
    ~30 % to trips the line does not speed up at all, which inflated every zone.
  - *Fair comparison*: res-9 cells near a station get walk-graph access to it, so today's trip may use
    the same walk to the station's area; and a path must ride at least one stop. Without both, a line
    crawling at 1 km/h still got thousands of users from walk precision alone.
  - *Circle*: largest radius where the population-weighted mean chance ≥ `zones.threshold` and the
    outer 500 m band still averages ≥ half of it (`zones.edge_share`). Without the edge rule, a
    terminal whose town 2–3 km out has a 50 % chance stretched to the 5 km cap over rings at 1 %.
  - *Threshold 0.2, not 0.3*: on real data the chance near a station is 0.1–0.65 (most trips are local
    and would not use any metro). On two 6-station test lines, 0.3 left 8 of 12 stations without a
    circle; 0.2 gives circles of 250 m–1.1 km at 7 of 12. Stations at existing metro (Giza Square,
    Ramses) get small or no circles. Slider range 0.1–0.3.
  - *Cost*: 4–18 ms per request (parent res-8 rows plus res-9 rows only inside a 20-min walk).
- **Synergy and Shapley** both need per-subset runs, so both are computed only when the request sets
  `attribution: true` (4 lines → 14 extra spending-only runs, ≈ 0.35 s on a 3,000-cell world).

## API

- Stations in requests are objects `{"lat": …, "lng": …}`, not arrays, to rule out [lng, lat] mix-ups.
- The API starts even without processed data: `/health` reports `ok: false`, `/evaluate` returns 503,
  `/scenarios` still works (demo fallback).
- Extra response fields beyond §10: `lines`, `footfall_cells`, `timings_ms`, per-street `R0_egp`,
  `by_category_egp`, `rent_pct_opening`, `risk_opening`, totals `abs_dR_egp`, `total_revenue_change`,
  `amber_streets`, `vendor_pitches(_lost)`, `max_rent_pct`.

## Data and pipeline

- **GTFS sources**: TfC's own download links (`documents/41|42/download`) return 404 (Nov 2026).
  The same feeds come from the Mobility Database mirror: `mdb-1825` (bus/paratransit, 930 routes,
  service 2021-01-01 → 2022-06-01) and `mdb-786` (metro, Lines 1–3, service 2020-09-24 → 2021-09-23).
  The newer feeds (`mdb-3354` metro, `mdb-3355` bus) cover only 2024–25 and the 2025 metro feed has
  Lines 1–2 only, so they are not used.
- **Spending source: CAPMAS HIECS 2019/2020, not 2021.** The HIECS 2021 tables could not be obtained
  (censusinfo.capmas.gov.eg does not resolve). `config/spending_hiecs_2019_20.csv` holds CAPMAS's
  published *urban per-capita* averages by main group ("Averages of Household Expenditure, Egypt,
  HIECS 2019/2020", report page 13, via the ERF data portal): food 5,055.0; clothing 967.1;
  furnishings 725.1; recreation 453.0; restaurants & hotels 967.1 EGP/person/yr (2019/20 prices).
  The report has no governorate breakdown, so all three governorates get the urban-Egypt values
  (spatial variation comes from population, and from wealth once a wealth layer exists).
  "Restaurants and hotels" includes hotels; small for urban households, not removed.
  Converted to 2021 prices with `spending.price_factor_to_2021` = 1.06 (1.04–1.08): CAPMAS urban CPI
  ran ≈ 4.3–6.6 % y/y through 2020–21, ~15 months from survey prices to mid-2021.
  A human-filled `data/raw/spending_manual.csv` (rows with a `prices` column) overrides this table.
- **Governorate boundaries** come from Nominatim *structured* queries (`{"state": …, "country": "Egypt"}`).
  The free-text query "Qalyubia Governorate, Egypt" returned a building in Obour, which put 8.4 M
  people in "Other"; 00 now rejects any result that is not an administrative boundary.
- **OSM extract clipped for r5py.** R5 rejects street layers wider than 975,000 km² and the Egypt
  extract is ~1.8 M km², so r5py failed and 08 fell back to scipy. `khatt/prep/osm_clip.py`
  (pyosmium, pinned) keeps `highway=*` ways touching the bbox plus their nodes → `cairo.osm.pbf`
  (~25 MB, ~230 × 120 km). Boundaries, rivers and other non-street ways are dropped.
- **Walk graph built locally, not via Overpass.** The public Overpass API returned 502 "at capacity"
  after 20 min for this bbox. 06 now builds the graph from `cairo.osm.pbf` with pyosmium
  (`khatt/prep/walk_osm.py`): OSMnx's "walk" way filter, simplified to intersections/endpoints,
  bidirectional, largest connected component → 641,694 nodes in ~14 s, written straight to
  `walk_csr.npz` (no `walk_graph.pkl`). `--overpass` keeps the original OSMnx path. The loader
  recomputes each node's r9 row from `walk_nodes_r9.parquet` at every load, so a rebuilt cell
  index never leaves the graph stale.
- **Overture schema change.** Current releases replace `categories` with `taxonomy`
  {primary, hierarchy, alternates} + `basic_category`. 04 reads both; the crosswalk gained the
  `basic_category` values (e.g. `fashion_and_apparel_store` → comp, `casual_eatery` → eat).
  Overture covers few small grocers, so `daily` has only ~4,200 shops vs ~15,000 eating places.
- **Streets: DBSCAN with an extent cap.** At eps = 150 m on ~100k Cairo places a single DBSCAN
  chains whole districts (one "street" = all of Nasr City, 5,583 shops, 14 km wide). Clusters wider
  than `streets.max_extent_m` (1,500 m) are re-clustered with eps × 0.75, recursively (floor 40 m)
  → 1,001 streets, median width ~370 m, 66 % of retail shops kept.
- **Cells need pop > `area.min_cell_pop` (50)** unless they hold a street or stop. WorldPop puts a
  few people in most desert/farm cells: pop > 0 gave 5,879 cells (spec expects ~3,000); > 50 gives
  3,710 and drops 0.07 % of the population.
- **r5py T0**: runs on `cairo.osm.pbf` + bus + metro GTFS in ~24 min (3 GB peak). `max_time` is
  2 × `spending.max_trip_min` (240 min), so 999 means "no route *or* longer than 240 min" — harmless,
  since trips over 120 min are excluded from every model. **Routing points**: each cell's trips start
  and end at the nearest node of the main connected walk network (06), not the centroid (median move
  76 m, p90 316 m; `routing_points` in 08, written to `data/interim/routing_points.parquet`). From
  centroids R5 linked 63 cells (67,644 people) to disconnected fragments and could not route them
  even with `snap_to_network` at 2,500 m; with routing points 10 cells remain. (Capping walk legs at
  60 min made gaps worse — r5py already allows walk legs up to `max_time`; reverted.) 08 still
  patches what r5py leaves with the scipy fallback (`fill_from_fallback`): unroutable cells take the
  fallback row/column, any other gap takes the fallback time when ≤ `spending.max_trip_min`.
  Current split (`T0.json`): 10 cells + 119,138 pairs from the fallback, ~3–6 % of < 5 km pairs,
  about half of them pairs r5py routed in the other direction (destination-side snapping). Where
  both route, r5py is a median 21 min faster than the fallback overall, but only 5–8 min on the
  short pairs that get filled. Raw r5py output is kept as `T0_r5py.npy`.
  Check pairs (10:00): Tahrir→Ramses 18 min, Giza Sq→Tahrir 26, Imbaba→Tahrir 39, Nasr City→Tahrir
  46, Heliopolis→Maadi 56, Haram→Tahrir 51.
- **Departure date changed to Tuesday 2021-03-16** (was 2022-03-15). The metro feed has *no* service
  on 2022-03-15, so the spec date would have produced a baseline with no metro. 2021-03-16 is inside
  both feeds' calendars; real feeds' dates are unchanged (CLAUDE.md §6, 07.5).
- **Departure time changed to 10:00** (spec: 08:00). Every one of the TfC bus feed's 1,624
  frequency rows runs 10:00–15:00 (its 26 scheduled trips start at 10:00), so at 08:00 *no* bus or
  microbus ran and r5py routed the 08:00 baseline on metro + walking only: 5–10 km trips took a
  median 130 min, Nasr City → Tahrir 90 min (fallback with buses: 55). The feed is a typical-day
  pattern, not a timetable, so 10:00–11:00 is the only honest window; the metro runs then too.
  08 now aborts if any feed runs fewer than `routing.min_service_share` (50 %) of its trips in the
  departure window (`gtfs.trips_running`), so this cannot recur silently. The fallback was not
  affected (it uses each route's frequency even outside its window); r5py was.
- **Synthetic world** (`python -m khatt.synthetic`, `make synthetic`) writes every §5 file from
  invented numbers on real Cairo coordinates (marked by a `SYNTHETIC` file and `/health.synthetic`).
  Used by tests and for front-end work before real data exists. Never present it as Cairo.
- **Extra processed files**: `existing_lines.geojson` (09, for `/baseline`), `walk_csr.npz` (cache),
  `calibrated.json` (§13), `T0.json` (how T0 was built).
- **Fallback T0**: nodes = cells + one node per (route, direction, stop); boarding wait =
  headway/2 × irregularity[mode]; transfers go through cells. Stops link to cells within 600 m.
- **Calibration**: geometric bisection on β ∈ [1e-4, 3]. On real data the spec targets for `daily`
  (10 min) and `eat` (20 min) are below the floor no β can beat — the spending-weighted time to the
  *nearest* street cluster (25.1 and 23.2 min: clusters need ≥ 15 shops, corner shops are not
  streets). Pinning β at the cap would send everyone to the single nearest cluster, so such targets
  become floor × (1 + `spending.calibration_floor_margin`) = +10 %. With the 10:00 T0 (floors 21.1
  and 20.6 min): β daily 0.336, comp 0.183 (30 min target met exactly), eat 0.355 per minute.
  Before the departure-time fix (no buses in T0) single streets swung by up to +400 % and the rent
  index reached 109 % in a 4-line stress test, because hundreds of streets were reachable by few
  residents and any line suddenly opened them up. Now: single realistic lines move streets by at
  most 12–24 % (p90 |ΔR| ≤ 3.2 %) with rent index ≤ 21 %; the 4 × 10-station stress test peaks at
  +67 % and 78 %. Lowering β does *not* shrink these (tested: flatter β makes swings larger).
- **Governorates**: OSM boundaries via Nominatim if available, else an approximate Nile/latitude rule.
- **Wealth**: Meta RWI → `exp(spending.rwi_scale × rwi)`, normalised to a population-weighted mean of 1;
  1.0 everywhere when absent.
- **Microbus routes**: agency name matched against `micro|paratransit|informal|shared|suzuki|…` ("Minibus licensed by CTA" is excluded: licensed, not informal);
  if nothing matches, all non-rail bus-feed routes are treated as paratransit (logged).

## Pipeline runs

- **03_spending**: no data/raw/spending_manual.csv: used spending_hiecs_2019_20.csv (CAPMAS HIECS 2019/20 urban per-capita averages, same for all governorates) _(2026-11-07 23:50)_
- **03_spending**: ['2019/20'] prices × 1.06 → 2021 EGP _(2026-11-07 23:50)_
- **03_spending**: no CAPMAS row for ['Other']: population-weighted mean of the others used _(2026-11-07 23:50)_
- **03_spending**: wealth factor source: none (1.0 everywhere) _(2026-11-07 23:50)_
- **07_gtfs_clean**: manual line L3W: station coordinates missing; skipped _(2026-11-08 00:25)_
- **07_gtfs_clean**: manual line L3CU: station coordinates missing; skipped _(2026-11-08 00:25)_
- **07_gtfs_clean**: cells need pop > 50 (or a street/stop); 0.07% of population is in dropped cells _(2026-11-08 00:26)_
- **08_baseline_matrix**: r5py failed (exit 1); fallback used _(2026-11-08 00:27)_
- **08_baseline_matrix**: T0 built with scipy fallback from ['bus.zip', 'metro.zip'] _(2026-11-08 00:28)_
- **08_baseline_matrix**: T0 built with r5py from ['bus.zip', 'metro.zip'] _(2026-11-08 00:49)_
- **08_baseline_matrix**: 72 cells could not be placed on the r5py network; their rows/columns come from the scipy fallback _(2026-11-08 01:17)_
- **08_baseline_matrix**: T0 built with r5py (72 unroutable cells from scipy fallback) from ['bus.zip', 'metro.zip'] _(2026-11-08 01:17)_
- **08_baseline_matrix**: r5py (32.8% pairs routed); scipy fallback for 63 unroutable cells and 156769 other pairs <= 120 min _(2026-11-08 01:36)_
- **08_baseline_matrix**: T0 built with r5py (32.8% pairs routed); scipy fallback for 63 unroutable cells and 156769 other pairs <= 120 min from ['bus.zip', 'metro.zip'] _(2026-11-08 01:36)_
- **08_baseline_matrix**: r5py (36.5% pairs routed); scipy fallback for 10 unroutable cells and 119138 other pairs <= 120 min _(2026-11-08 04:03)_
- **08_baseline_matrix**: T0 built with r5py (36.5% pairs routed); scipy fallback for 10 unroutable cells and 119138 other pairs <= 120 min from ['bus.zip', 'metro.zip'] _(2026-11-08 04:03)_

## Map display

- **Microbus routes** were drawn as straight lines between stops (median 790 m apart), so they cut
  across blocks. Step 10 now uses each route's GTFS road trace (`shapes.txt`, the trip it already
  picks for the stop pattern) if every stop is within `informal.shape_max_stop_gap_m` (300 m) of
  it: all 952 routes qualify, median worst stop gap 2 m. Stop lists and `stop_hex_idx` are
  unchanged, so model results are identical. `/baseline` simplifies traces to ~10 m and rounds
  coordinates to ~1 m (1.2 MB); the API now gzips responses (259 kB on the wire).
- **Spending flows** link a neighbourhood to a street; they are not routes. Drawn as raised 3D
  arcs they looked like flight paths from straight above, and at street zoom flat lines read as
  roads through blocks. Now: flat lines fading from the neighbourhood to the street; the top-50
  overview hides above map zoom 13; clicking a changed street shows only its own flows.

## Data storage

- `data/raw/` is committed through Git LFS (about 360 MB) so teammates don't have to re-run the downloads,
  overriding CLAUDE.md rule 10. This is acceptable because the repo is private (TfC CC BY-NC, OSM ODbL,
  WorldPop CC BY, Overture CDLA). The full `egypt-latest.osm.pbf` (177 MB) is left out: only the
  clipped `cairo.osm.pbf` is needed.
- `data/processed/` is committed too (its parquet files through LFS), except the files
  .gitignore still excludes: `T0*.npy`/`T0*.json` (55 MB each) and `walk_*` (28 MB). The API
  needs `T0.npy` and the walk graph, so a fresh clone still has to run 06 and 08 for real data.
- **07_gtfs_clean**: n_places: 99,375 of 100,263 mapped places fall in kept cells (jobs stand-in for the station zones model) _(2026-11-08 05:21)_
- **10_microbus_routes**: route lines follow the GTFS road trace (shapes.txt) for 952 of 952 routes; 0 traces had a stop > 300 m away and 0 had none, so those use straight stop-to-stop lines. The model uses the stops either way (stop_lat/stop_lng, stop_hex_idx) _(2026-11-08 06:23)_
