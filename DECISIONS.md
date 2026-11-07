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
- **Model 2 usefulness**: share of the catchment's spending-weighted trips *to shopping streets*
  with T1 < T0 − 0.5 (scenario probabilities), weighted by catchment population of each parent cell.
- **Model 2 pass-by**: walk distance from the station's catchment Dijkstra to each street's nearest
  walk node (street within 400 m straight line first). Heatmap (`footfall_cells`) = Σ station exits ×
  e^(−walk_m / passby_decay_m) over each station's reached r9 cells.
- **Model 3**: added `rents.footfall_term` (assumption, central 0.1) for the spec's
  `footfall_term × passby_share`. Street rent uses its own pass-by share; hex rent uses Σ pass-by /
  Σ R0 of its streets. Risk follows the spec literally, so a street losing even 0.05 % with no rent
  change is *amber* (profit signal < 0). On synthetic data the rent index has a long tail (p99 ≈ 60 %)
  in cells with poor baseline access — review against real data before the demo.
- **Model 4**: consecutive duplicate stop hexes are collapsed before forming pairs (otherwise pairs
  inside one cell dilute `share_at_risk`). Feeder = any route vertex within 400 m of a new station.
  Vendor pitches lost = Σ_competing share_at_risk × `informal.vendors_per_route` (new parameter,
  assumption: 6), labelled an estimate.
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
- **Departure date changed to Tuesday 2021-03-16** (was 2022-03-15). The metro feed has *no* service
  on 2022-03-15, so the spec date would have produced a baseline with no metro. 2021-03-16 is inside
  both feeds' calendars; real feeds' dates are unchanged (CLAUDE.md §6, 07.5).
- **Synthetic world** (`python -m khatt.synthetic`, `make synthetic`) writes every §5 file from
  invented numbers on real Cairo coordinates (marked by a `SYNTHETIC` file and `/health.synthetic`).
  Used by tests and for front-end work before real data exists. Never present it as Cairo.
- **Extra processed files**: `existing_lines.geojson` (09, for `/baseline`), `walk_csr.npz` (cache),
  `calibrated.json` (§13), `T0.json` (how T0 was built).
- **Fallback T0**: nodes = cells + one node per (route, direction, stop); boarding wait =
  headway/2 × irregularity[mode]; transfers go through cells. Stops link to cells within 600 m.
- **Calibration**: geometric bisection on β ∈ [1e-4, 3]. If a target is unreachable, β sits at the
  bound and a note is written (happens for `daily` on the synthetic world: only ~300 streets).
- **Governorates**: OSM boundaries via Nominatim if available, else an approximate Nile/latitude rule.
- **Wealth**: Meta RWI → `exp(spending.rwi_scale × rwi)`, normalised to a population-weighted mean of 1;
  1.0 everywhere when absent.
- **Microbus routes**: agency name matched against `micro|paratransit|informal|shared|minibus|…`;
  if nothing matches, all non-rail bus-feed routes are treated as paratransit (logged).

## Pipeline runs

- **01_grid**: No governorate boundaries: used the approximate Nile/latitude rule. _(2026-11-07 22:58)_
- **00_download**: osmium not installed: r5py uses the full Egypt OSM extract (more memory). `brew install osmium-tool` to clip to the bbox. _(2026-11-07 23:11)_
