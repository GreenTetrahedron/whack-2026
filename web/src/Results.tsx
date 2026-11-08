import { useState, type ReactNode } from "react";
import type { LayerToggles } from "./MapView";
import type { DraftLine, EvalResult, StreetResult } from "./types";
import { egp, num, pct, RISK_COLOR } from "./format";

// What each timed step in khatt/evaluate.py does, in plain words.
const STEPS: { key: string; label: string; what: string }[] = [
  { key: "engine_setup", label: "Line graph", what: "Snap stations to cells, ride times, best paths between all stations incl. transfers (Floyd–Warshall)" },
  { key: "overlay", label: "Travel times", what: "New door-to-door times T1 = min(today, walk → ride the new line → walk) for every origin cell (numba)" },
  { key: "spending", label: "Spending", what: "Huff model: re-split each neighbourhood’s shopping budget across streets using the new travel times" },
  { key: "footfall", label: "Footfall", what: "Riders per new station from its 10-min walk catchment, then pass-by spending on nearby streets" },
  { key: "rents", label: "Rents", what: "Market potential near stations → rent index; compare with revenue change → margin-squeeze risk" },
  { key: "informal", label: "Microbuses", what: "Which microbus trips get faster by metro (competing) or connect to it (feeder); vendor pitches" },
];
const STEP_COLORS = ["#6c5ce7", "#0984e3", "#00b894", "#fdcb6e", "#e17055", "#d63031"];

type Tab = "spending" | "footfall" | "rents" | "microbus" | "lines";

export default function Results(props: {
  result: EvalResult;
  roundTripMs: number;
  stale: boolean;
  lines: DraftLine[];
  view: "before" | "after";
  setView: (v: "before" | "after") => void;
  toggles: LayerToggles;
  setToggles: (t: LayerToggles) => void;
}) {
  const { result, toggles, setToggles } = props;
  const [tab, setTab] = useState<Tab>("spending");
  const t = result.totals;
  const toggle = (k: keyof LayerToggles) => setToggles({ ...toggles, [k]: !toggles[k] });

  return (
    <>
      <section>
        <h2>3 · What the backend did</h2>
        {props.stale && <div className="warnbox">The lines or assumptions changed since this run — press Construct again.</div>}
        <Timings result={result} roundTripMs={props.roundTripMs} />
        {result.warnings.length > 0 && (
          <ul className="warnings">
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2>4 · Results</h2>
        <div className="row">
          <div className="seg">
            <button className={props.view === "before" ? "on" : ""} onClick={() => props.setView("before")}>
              Before
            </button>
            <button className={props.view === "after" ? "on" : ""} onClick={() => props.setView("after")}>
              After
            </button>
          </div>
          <div className="layer-toggles">
            <Check on={toggles.flows} onClick={() => toggle("flows")} label="Flows" />
            <Check on={toggles.rent} onClick={() => toggle("rent")} label="Rent hexes" />
            <Check on={toggles.footfall} onClick={() => toggle("footfall")} label="Footfall" />
            <Check on={toggles.microbus} onClick={() => toggle("microbus")} label="Microbus" />
            <Check on={toggles.allMicrobus} onClick={() => toggle("allMicrobus")} label="All routes" />
          </div>
        </div>
        <div className="tabs">
          {(
            [
              ["spending", "Spending"],
              ["footfall", "Footfall"],
              ["rents", "Rents"],
              ["microbus", "Microbuses"],
              ["lines", "Lines"],
            ] as [Tab, string][]
          ).map(([k, label]) => (
            <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
              {label}
            </button>
          ))}
        </div>

        {tab === "spending" && (
          <>
            <Headlines
              items={[
                [egp(t.abs_dR_egp / 2), "spending moved between streets / yr"],
                [num(t.streets_gaining), "streets gain"],
                [num(t.streets_losing), `streets lose (${t.local_losers} near a new station)`],
              ]}
            />
            <p className="explain">
              Total spending is fixed, so it only moves: streets the new line makes easier to reach take customers from
              streets it doesn’t. Gains and losses sum to ≈ 0 ({egp(t.dR_egp, true)}).
            </p>
            <StreetTable
              title="Biggest winners"
              rows={[...result.streets].filter((s) => s.dR_egp > 0).sort((a, b) => b.dR_egp - a.dR_egp).slice(0, 8)}
            />
            <StreetTable
              title="Biggest losers"
              rows={[...result.streets].filter((s) => s.dR_egp < 0).sort((a, b) => a.dR_egp - b.dR_egp).slice(0, 8)}
            />
          </>
        )}

        {tab === "footfall" && (
          <>
            <Headlines
              items={[
                [num(t.riders_daily), "new riders / day"],
                [egp(t.passby_egp), "pass-by spending / yr"],
                [num(t.vendor_pitches), "street-vendor pitches at stations"],
              ]}
            />
            <p className="explain">
              Riders = residents within a 10-minute walk × Cairo’s riders-per-resident × how useful the line is to them.
              People walking to and from stations buy things on streets they pass.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Station</th>
                  <th className="r">Riders/day</th>
                  <th className="r">Residents ≤10 min</th>
                  <th className="r">Usefulness</th>
                </tr>
              </thead>
              <tbody>
                {result.stations.map((s) => (
                  <tr key={`${s.line_id}${s.idx}`}>
                    <td>
                      <Swatch lines={props.lines} id={s.line_id} /> {s.line_id}
                      {s.idx + 1}
                    </td>
                    <td className="r">{num(s.riders_daily)}</td>
                    <td className="r">{num(s.catchment_pop10)}</td>
                    <td className="r">{s.usefulness.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {tab === "rents" && (
          <>
            <Headlines
              items={[
                [pct(t.max_rent_pct), "highest rent index rise"],
                [num(t.red_streets), `red streets (${t.amber_streets} amber)`],
                [num(t.unbranded_at_risk), "unbranded shops at risk"],
              ]}
            />
            <p className="explain">
              Rents are an <b>index</b>, not EGP. A street is red when its rent rise outweighs its revenue change. Turn on
              “Rent hexes” to see the rent index on the map.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Street</th>
                  <th className="r">Rent idx</th>
                  <th className="r">Revenue</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {[...result.streets]
                  .filter((s) => s.risk !== "green")
                  .sort((a, b) => (a.risk === b.risk ? b.rent_pct - a.rent_pct : a.risk === "red" ? -1 : 1))
                  .slice(0, 12)
                  .map((s) => (
                    <tr key={s.street_id}>
                      <td className="name">{s.name}</td>
                      <td className="r">{pct(s.rent_pct)}</td>
                      <td className="r">{pct(s.revenue_pct)}</td>
                      <td style={{ color: RISK_COLOR[s.risk] }}>{s.risk}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </>
        )}

        {tab === "microbus" && (
          <>
            <Headlines
              items={[
                [num(t.microbus.competing), "routes competing (orange)"],
                [num(t.microbus.feeder), "routes feeding the line (green)"],
                [num(t.vendor_pitches_lost), "vendor pitches lost (estimate)"],
              ]}
            />
            <p className="explain">
              A microbus route competes when ≥ 30 % of its stop-to-stop trips become 5+ minutes faster by the new line.
              {` ${num(t.microbus.unaffected)} routes are unaffected.`}
            </p>
            <table>
              <thead>
                <tr>
                  <th>Route</th>
                  <th>Status</th>
                  <th className="r">Trips that switch</th>
                </tr>
              </thead>
              <tbody>
                {[...result.microbus_routes]
                  .sort((a, b) => b.share_at_risk - a.share_at_risk)
                  .slice(0, 12)
                  .map((r) => (
                    <tr key={r.route_id}>
                      <td className="name" title={r.route_id}>
                        {r.route_id.replace(/^bus:/, "").slice(0, 14)}
                      </td>
                      <td style={{ color: r.status === "competing" ? "#e07000" : "#28a046" }}>{r.status}</td>
                      <td className="r">{Math.round(100 * r.share_at_risk)}%</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </>
        )}

        {tab === "lines" && (
          <>
            <table>
              <thead>
                <tr>
                  <th>Line</th>
                  <th className="r">Stations</th>
                  <th className="r">km</th>
                  <th className="r">Headway</th>
                  <th className="r">Speed</th>
                </tr>
              </thead>
              <tbody>
                {result.lines.map((l) => (
                  <tr key={l.line_id}>
                    <td>
                      <Swatch lines={props.lines} id={l.line_id} /> {l.name}
                    </td>
                    <td className="r">{l.n_stations}</td>
                    <td className="r">{l.length_km.toFixed(1)}</td>
                    <td className="r">{l.headway_min} min</td>
                    <td className="r">{l.speed_kmh} km/h</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.synergy.length > 0 ? (
              <>
                <p className="explain">
                  Synergy = effect of both lines together minus each alone. Positive means the pair does more than the sum of
                  its parts (e.g. via an interchange); negative means they partly serve the same trips.
                </p>
                <table>
                  <tbody>
                    {result.synergy.map((s) => (
                      <tr key={s.lines.join()}>
                        <td>{s.lines.join(" + ")}</td>
                        <td className="r">{egp(s.abs_dR_egp, true)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            ) : (
              <p className="explain">
                Draw a second line and tick “Split the effect between lines” to see synergy and each line’s share.
              </p>
            )}
          </>
        )}
      </section>
    </>
  );
}

function Timings({ result, roundTripMs }: { result: EvalResult; roundTripMs: number }) {
  const steps = STEPS.filter((s) => result.timings_ms[s.key] !== undefined);
  const total = steps.reduce((a, s) => a + result.timings_ms[s.key], 0) || 1;
  return (
    <div className="timings">
      <div className="time-head">
        <b>{Math.round(result.elapsed_ms)} ms</b> on the server
        <span className="muted"> · {Math.round(roundTripMs)} ms incl. network</span>
        <span className={result.elapsed_ms < 2000 ? "ok" : "bad"}> {result.elapsed_ms < 2000 ? "✓ under 2 s" : "over 2 s"}</span>
      </div>
      <div className="stack">
        {steps.map((s, i) => (
          <div
            key={s.key}
            style={{ width: `${(100 * result.timings_ms[s.key]) / total}%`, background: STEP_COLORS[i] }}
            title={`${s.label}: ${result.timings_ms[s.key].toFixed(0)} ms`}
          />
        ))}
      </div>
      <ol className="steps">
        {steps.map((s, i) => (
          <li key={s.key}>
            <span className="dot" style={{ background: STEP_COLORS[i] }} />
            <div>
              <b>{s.label}</b> <span className="muted">{result.timings_ms[s.key].toFixed(0)} ms</span>
              <div className="small muted">{s.what}</div>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Headlines({ items }: { items: [string, string][] }) {
  return (
    <div className="headlines">
      {items.map(([v, l]) => (
        <div key={l}>
          <div className="big">{v}</div>
          <div className="small muted">{l}</div>
        </div>
      ))}
    </div>
  );
}

function StreetTable({ title, rows }: { title: string; rows: StreetResult[] }) {
  if (!rows.length) return null;
  return (
    <table>
      <thead>
        <tr>
          <th>{title}</th>
          <th className="r">EGP / yr</th>
          <th className="r">%</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((s) => (
          <tr key={s.street_id}>
            <td className="name" title={s.name}>
              {s.name}
              {s.local_loser && <span className="tag">near station</span>}
            </td>
            <td className="r">{egp(s.dR_egp, true)}</td>
            <td className="r" style={{ color: s.dR_pct < 0 ? "#c62828" : "#1f5fa8" }}>
              {pct(s.dR_pct)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Check({ on, onClick, label }: { on: boolean; onClick: () => void; label: ReactNode }) {
  return (
    <label className="check inline">
      <input type="checkbox" checked={on} onChange={onClick} /> {label}
    </label>
  );
}

function Swatch({ lines, id }: { lines: DraftLine[]; id: string }) {
  const c = lines.find((l) => l.id === id)?.color ?? "#999";
  return <span className="swatch sm" style={{ background: c }} />;
}
