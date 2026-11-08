import { useCallback, useEffect, useMemo, useState } from "react";
import MapView, { type LayerToggles } from "./MapView";
import Results from "./Results";
import { API_URL, getJSON, postJSON } from "./api";
import type { Baseline, DraftLine, EvalResult, Health, LatLng, ParamRange } from "./types";
import { num } from "./format";

const LINE_IDS = ["A", "B", "C", "D"]; // spec: up to 4 lines
const LINE_COLORS = ["#7b2cbf", "#0f9d8a", "#d63384", "#c77d00"];

// A rough Giza Square → Pyramids corridor so there is something to evaluate with one click.
const EXAMPLE: LatLng[] = [
  { lat: 30.0108, lng: 31.2075 },
  { lat: 30.0062, lng: 31.1925 },
  { lat: 30.0016, lng: 31.1775 },
  { lat: 29.9968, lng: 31.1625 },
  { lat: 29.9921, lng: 31.1475 },
  { lat: 29.9878, lng: 31.1335 },
];

function newLine(i: number, defaults: { headway: number; speed: number }): DraftLine {
  return {
    id: LINE_IDS[i],
    name: `Line ${LINE_IDS[i]}`,
    color: LINE_COLORS[i],
    stations: [],
    headway_min: defaults.headway,
    speed_kmh: defaults.speed,
    visible: true,
  };
}

function km(stations: LatLng[]): number {
  let d = 0;
  for (let i = 1; i < stations.length; i++) {
    const a = stations[i - 1];
    const b = stations[i];
    const rad = Math.PI / 180;
    const x = (b.lng - a.lng) * rad * Math.cos(((a.lat + b.lat) / 2) * rad);
    const y = (b.lat - a.lat) * rad;
    d += 6371 * Math.hypot(x, y);
  }
  return d;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<Baseline | null>(null);
  const [params, setParams] = useState<ParamRange[]>([]);
  const [sliders, setSliders] = useState<string[]>([]);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [defaults, setDefaults] = useState({ headway: 4, speed: 35 });

  const [lines, setLines] = useState<DraftLine[]>([newLine(0, { headway: 4, speed: 35 })]);
  const [activeId, setActiveId] = useState("A");
  const [drawing, setDrawing] = useState(true);
  const [attribution, setAttribution] = useState(false);

  const [result, setResult] = useState<EvalResult | null>(null);
  const [evaluatedKey, setEvaluatedKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  const [view, setView] = useState<"before" | "after">("after");
  const [toggles, setToggles] = useState<LayerToggles>({
    flows: false, // a street's flows show when it is clicked; this is the all-streets overview
    rent: false,
    footfall: false,
    microbus: true,
    allMicrobus: false,
    zones: true,
  });

  // ---- load what the backend has precomputed ----
  const connect = useCallback(async () => {
    setApiError(null);
    try {
      const h = await getJSON<Health>("/health");
      setHealth(h);
      if (!h.ok) return;
      const [b, p] = await Promise.all([
        getJSON<Baseline>("/baseline"),
        getJSON<{ params: ParamRange[]; sliders: string[] }>("/params"),
      ]);
      setBaseline(b);
      setParams(p.params);
      setSliders(p.sliders);
      const central = (k: string) => p.params.find((x) => x.key === k)?.central;
      const d = { headway: central("line.headway_min") ?? 4, speed: central("line.speed_kmh") ?? 35 };
      setDefaults(d);
      setLines((ls) => ls.map((l) => (l.stations.length ? l : { ...l, headway_min: d.headway, speed_kmh: d.speed })));
    } catch (e) {
      setApiError(String(e instanceof Error ? e.message : e));
    }
  }, []);
  useEffect(() => {
    connect();
  }, [connect]);

  // ---- line editing ----
  const updateLine = (id: string, f: (l: DraftLine) => DraftLine) =>
    setLines((ls) => ls.map((l) => (l.id === id ? f(l) : l)));

  const addStation = useCallback(
    (lat: number, lng: number) => updateLine(activeId, (l) => ({ ...l, stations: [...l.stations, { lat, lng }] })),
    [activeId],
  );
  const removeStation = useCallback(
    (id: string, idx: number) => updateLine(id, (l) => ({ ...l, stations: l.stations.filter((_, i) => i !== idx) })),
    [],
  );
  const addLine = () => {
    const free = LINE_IDS.findIndex((id) => !lines.some((l) => l.id === id));
    if (free < 0) return;
    setLines((ls) => [...ls, newLine(free, defaults)].sort((a, b) => a.id.localeCompare(b.id)));
    setActiveId(LINE_IDS[free]);
    setDrawing(true);
  };
  const deleteLine = (id: string) => {
    const rest = lines.filter((l) => l.id !== id);
    if (!rest.length) {
      setLines([newLine(0, defaults)]);
      setActiveId("A");
    } else {
      setLines(rest);
      if (activeId === id) setActiveId(rest[0].id);
    }
  };
  const loadExample = () => {
    updateLine(activeId, (l) => ({ ...l, stations: EXAMPLE }));
    setDrawing(false);
  };

  const ready = lines.filter((l) => l.stations.length >= 2);
  const requestBody = useMemo(
    () => ({
      lines: ready.map((l) => ({
        line_id: l.id,
        name: l.name,
        stations: l.stations,
        headway_min: l.headway_min,
        speed_kmh: l.speed_kmh,
      })),
      params: Object.keys(overrides).length ? overrides : undefined,
      attribution: attribution && ready.length > 1,
    }),
    [ready, overrides, attribution],
  );
  const requestKey = JSON.stringify(requestBody);
  const stale = result !== null && requestKey !== evaluatedKey;

  const construct = async () => {
    if (!ready.length) {
      setEvalError("Add at least 2 stations to a line first.");
      return;
    }
    setLoading(true);
    setEvalError(null);
    try {
      const res = await postJSON<EvalResult>("/evaluate", requestBody);
      setResult(res);
      setEvaluatedKey(requestKey);
      setView("after");
      setDrawing(false);
    } catch (e) {
      setEvalError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  };

  const active = lines.find((l) => l.id === activeId) ?? lines[0];

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <h1>
            Khatt <span className="ar">خط</span>
          </h1>
          <p className="sub">Draw a metro line in Greater Cairo and see what it does to local businesses.</p>
          <Status health={health} apiError={apiError} baseline={baseline} onRetry={connect} />
        </header>

        <section>
          <h2>Draw your lines</h2>
          <div className="lines">
            {lines.map((l) => (
              <div
                key={l.id}
                className={`line-row ${l.id === activeId ? "active" : ""}`}
                onClick={() => setActiveId(l.id)}
              >
                <span className="swatch" style={{ background: l.color }} />
                <input
                  className="line-name"
                  value={l.name}
                  onChange={(e) => updateLine(l.id, (x) => ({ ...x, name: e.target.value }))}
                />
                <span className="muted small">
                  {l.stations.length} stn · {km(l.stations).toFixed(1)} km
                </span>
                <button
                  className="icon"
                  title={l.visible ? "Hide" : "Show"}
                  onClick={(e) => {
                    e.stopPropagation();
                    updateLine(l.id, (x) => ({ ...x, visible: !x.visible }));
                  }}
                >
                  {l.visible ? "◉" : "○"}
                </button>
                <button
                  className="icon"
                  title="Delete line"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteLine(l.id);
                  }}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          <div className="row">
            <button className={drawing ? "toggle on" : "toggle"} onClick={() => setDrawing(!drawing)}>
              {drawing ? "✏️ Drawing: click map to add stations" : "✏️ Drawing off"}
            </button>
          </div>
          <div className="row">
            <button onClick={() => updateLine(activeId, (l) => ({ ...l, stations: l.stations.slice(0, -1) }))}>
              Undo
            </button>
            <button onClick={() => updateLine(activeId, (l) => ({ ...l, stations: [] }))}>Clear</button>
            <button onClick={addLine} disabled={lines.length >= LINE_IDS.length}>
              + New line
            </button>
            <button onClick={loadExample} title="Giza Square → Pyramids, approximate">
              Example
            </button>
          </div>
          {active && (
            <div className="row inputs">
              <label>
                Headway (min)
                <input
                  type="number"
                  min={1}
                  max={30}
                  step={0.5}
                  value={active.headway_min}
                  onChange={(e) => updateLine(active.id, (l) => ({ ...l, headway_min: Number(e.target.value) || 1 }))}
                />
              </label>
              <label>
                Speed (km/h)
                <input
                  type="number"
                  min={10}
                  max={120}
                  step={1}
                  value={active.speed_kmh}
                  onChange={(e) => updateLine(active.id, (l) => ({ ...l, speed_kmh: Number(e.target.value) || 10 }))}
                />
              </label>
            </div>
          )}
          <p className="hint">Right-click a station to delete it. The active line is the highlighted one.</p>
        </section>

        <section>
          <h2>2 · Construct</h2>
          {ready.length > 1 && (
            <label className="check">
              <input type="checkbox" checked={attribution} onChange={(e) => setAttribution(e.target.checked)} />
              Split the effect between lines (synergy + Shapley, slower)
            </label>
          )}
          <Assumptions params={params} sliders={sliders} overrides={overrides} setOverrides={setOverrides} />
          <button className="primary" onClick={construct} disabled={loading || !health?.ok}>
            {loading ? "Evaluating…" : stale ? "Construct again (lines changed)" : "Construct line"}
          </button>
          {evalError && <div className="error">{evalError}</div>}
        </section>

        {result && (
          <Results
            result={result}
            stale={stale}
            lines={lines}
            view={view}
            setView={setView}
            toggles={toggles}
            setToggles={setToggles}
          />
        )}

        <footer>
          Transit data: Transport for Cairo (CC BY-NC 4.0) · Population: WorldPop (CC BY 4.0) · Places: Overture
          (CDLA-Permissive-2.0) · Map data © OpenStreetMap (ODbL). Model estimates, 2021 EGP prices.
        </footer>
      </aside>
      <main>
        <MapView
          baseline={baseline}
          result={result}
          lines={lines}
          activeId={activeId}
          drawing={drawing}
          view={view}
          toggles={toggles}
          onAddStation={addStation}
          onRemoveStation={removeStation}
        />
        <Legend hasResult={!!result && view === "after"} />
        {drawing && (
          <div className="map-hint">
            Adding stations to <b style={{ color: active?.color }}>{active?.name}</b> — click the map
          </div>
        )}
      </main>
    </div>
  );
}

function Status(props: { health: Health | null; apiError: string | null; baseline: Baseline | null; onRetry: () => void }) {
  const { health, apiError, baseline } = props;
  if (apiError)
    return (
      <div className="status bad">
        Can’t reach the API at {API_URL}. Start it with <code>make api</code>.{" "}
        <button className="link" onClick={props.onRetry}>
          Retry
        </button>
      </div>
    );
  if (!health) return <div className="status">Connecting to the API…</div>;
  if (!health.ok)
    return (
      <div className="status bad">
        API is up but has no data: {health.error}{" "}
        <button className="link" onClick={props.onRetry}>
          Retry
        </button>
      </div>
    );
  return (
    <div className={`status ${health.synthetic ? "warn" : "good"}`}>
      <b>{health.synthetic ? "Synthetic test world" : "Real Cairo data"}</b> · {num(health.n_hexes)} hex cells ·{" "}
      {num(health.n_streets)} shopping streets
      {baseline && (
        <>
          {" "}
          · {baseline.existing_stations.length} metro stations · {baseline.microbus_routes.features.length} microbus
          routes
        </>
      )}
      <div className="muted small">
        Travel-time matrix T0: {health.t0_shape.join(" × ")} · loaded in {health.load_s} s
      </div>
    </div>
  );
}

const SLIDER_LABELS: Record<string, string> = {
  "spending.beta.daily": "β daily needs (distance aversion)",
  "spending.beta.comp": "β comparison goods",
  "spending.beta.eat": "β eating out",
  "rents.elasticity": "Rent elasticity to access",
  "spending.induced_pct": "Induced spending",
  "zones.threshold": "Area of effect: mean chance inside",
  "zones.work_trip_share": "Area of effect: share of work trips",
};

function Assumptions(props: {
  params: ParamRange[];
  sliders: string[];
  overrides: Record<string, number>;
  setOverrides: (o: Record<string, number>) => void;
}) {
  const rows = props.sliders.map((k) => props.params.find((p) => p.key === k)).filter(Boolean) as ParamRange[];
  if (!rows.length) return null;
  return (
    <details className="assumptions">
      <summary>Assumptions {Object.keys(props.overrides).length ? `(${Object.keys(props.overrides).length} changed)` : ""}</summary>
      {rows.map((p) => {
        const v = props.overrides[p.key] ?? p.value;
        return (
          <label key={p.key} className="slider" title={p.source}>
            <span>
              {SLIDER_LABELS[p.key] ?? p.key} <b>{v.toPrecision(3)}</b>
            </span>
            <input
              type="range"
              min={p.low}
              max={p.high}
              step={(p.high - p.low) / 50 || 0.01}
              value={v}
              onChange={(e) => {
                const next = { ...props.overrides, [p.key]: Number(e.target.value) };
                if (Math.abs(next[p.key] - p.value) < 1e-9) delete next[p.key];
                props.setOverrides(next);
              }}
            />
          </label>
        );
      })}
      <button className="link" onClick={() => props.setOverrides({})}>
        Reset to central values
      </button>
    </details>
  );
}

function Legend({ hasResult }: { hasResult: boolean }) {
  return (
    <div className="legend">
      {hasResult ? (
        <>
          <div>
            <span className="dot" style={{ background: "rgb(214,64,69)" }} /> street loses spending
          </div>
          <div>
            <span className="dot" style={{ background: "rgb(33,113,181)" }} /> street gains
          </div>
          <div>
            <span className="dot ring" /> red ring = margin squeeze risk
          </div>
          <div>
            <span className="dot zone" /> station area of effect
          </div>
          <div>
            <span className="bar" style={{ background: "rgb(240,120,0)" }} /> microbus competing
          </div>
          <div>
            <span className="bar" style={{ background: "rgb(40,160,70)" }} /> microbus feeder
          </div>
        </>
      ) : (
        <div>
          <span className="dot" style={{ background: "rgb(120,120,120)" }} /> shopping street (size = spending)
        </div>
      )}
      <div>
        <span className="bar" style={{ background: "rgb(90,90,90)" }} /> existing metro
      </div>
    </div>
  );
}
