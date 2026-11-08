import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import type { Layer, PickingInfo } from "@deck.gl/core";
import { ArcLayer, GeoJsonLayer, IconLayer, PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import type { Baseline, DraftLine, EvalResult, StationResult, StreetResult } from "./types";
import { divergingColor, egp, GAIN, hexToRgb, LOSS, num, pct, RISK_COLOR, type RGBA, sequentialColor } from "./format";

const BASEMAP = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
const CAIRO = { lat: 30.04, lng: 31.24, zoom: 10.5 };
// Flows link a neighbourhood to a street; they are not routes. Past this zoom they read like roads
// cutting through blocks, so the overview flows hide (a clicked street still shows its own).
const FLOWS_MAX_ZOOM = 13;

export type LayerToggles = {
  flows: boolean;
  rent: boolean;
  footfall: boolean;
  microbus: boolean;
  allMicrobus: boolean;
  zones: boolean;
};

type Props = {
  baseline: Baseline | null;
  result: EvalResult | null;
  lines: DraftLine[];
  activeId: string;
  drawing: boolean;
  view: "before" | "after";
  toggles: LayerToggles;
  onAddStation: (lat: number, lng: number) => void;
  onRemoveStation: (lineId: string, idx: number) => void;
};

type StreetRow = {
  street_id: number;
  name: string;
  lat: number;
  lng: number;
  R0_total: number;
  res?: StreetResult;
};

type DrawnStation = {
  lineId: string;
  idx: number;
  lat: number;
  lng: number;
  color: string;
  riders?: number;
  zone?: StationResult;
  active: boolean;
};

type ZoneRow = StationResult & { color: string };

// Overview: a residents' area → a street, egp signed (− = they spend less there). A clicked
// street's moves (`between` set): from the street that loses the money to the one that gains it.
type FlowRow = {
  from: [number, number];
  to: [number, number];
  egp: number;
  between?: { fromName: string; toName: string; into: boolean }; // into = points at the clicked street
};

/** Arrowhead pointing east, 20 × 20 units, anchored `gap` units past its tip so it stops short of
 * the street it points at. Drawn as a mask so the layer colours it. */
function arrowIcon(gap: number) {
  const w = 20 + gap;
  return {
    id: `arrow-${gap}`,
    url: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${2 * w}" height="40" viewBox="0 0 ${w} 20">` +
        '<path d="M0 0 L20 10 L0 20 L5 10 Z" fill="#000"/></svg>',
    )}`,
    width: 2 * w,
    height: 40,
    anchorX: 2 * w,
    anchorY: 20,
    mask: true,
  };
}
const ARROW = arrowIcon(8); // tip just short of the street dot
// a clicked street's arrows are all 16 px (1 unit = 0.8 px): 18 units puts the tips just outside its 13 px ring
const ARROW_RING = arrowIcon(18);
const ARROW_RING_PX = 16;

/** Direction of a flow line at its street end, degrees counter-clockwise from east, on the
 * Web Mercator map (where the flat line is straight). */
function flowAngle(f: FlowRow): number {
  const mercY = (lat: number) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  const dx = ((f.to[0] - f.from[0]) * Math.PI) / 180;
  const dy = mercY(f.to[1]) - mercY(f.from[1]);
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

function streetRadius(R0: number): number {
  return 35 + 5 * Math.sqrt(Math.max(R0, 0) / 1e6); // metres
}

export default function MapView(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  // handlers read through a ref so the map listeners never go stale
  const live = useRef(props);
  live.current = props;
  const [zoom, setZoom] = useState(CAIRO.zoom);
  const [selected, setSelected] = useState<number | null>(null); // street_id whose flows are shown

  useEffect(() => {
    if (!container.current) return;
    const map = new maplibregl.Map({
      container: container.current,
      style: BASEMAP,
      center: [CAIRO.lng, CAIRO.lat],
      zoom: CAIRO.zoom,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
    map.addControl(overlay);
    map.on("click", (e) => {
      if (live.current.drawing) {
        live.current.onAddStation(e.lngLat.lat, e.lngLat.lng);
        return;
      }
      // not drawing: clicking a changed street shows where its spending moves; elsewhere clears
      const info = overlay.pickObject({ x: e.point.x, y: e.point.y, radius: 4, layerIds: ["streets"] });
      const st = info?.object as StreetRow | undefined;
      setSelected(st?.res?.flows?.length ? st.street_id : null);
    });
    map.on("zoomend", () => setZoom(map.getZoom()));
    setZoom(map.getZoom()); // a re-created map (e.g. hot reload) starts at the default zoom
    map.on("contextmenu", (e) => {
      const info = overlay.pickObject({ x: e.point.x, y: e.point.y, radius: 6, layerIds: ["drawn-stations"] });
      const st = info?.object as DrawnStation | undefined;
      if (st) live.current.onRemoveStation(st.lineId, st.idx);
    });
    // MapLibre only listens to window resizes; follow the container instead (sidebar, first layout).
    const ro = new ResizeObserver(() => map.resize());
    ro.observe(container.current);
    mapRef.current = map;
    overlayRef.current = overlay;
    return () => {
      ro.disconnect();
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
  }, []);

  useEffect(() => {
    const canvas = mapRef.current?.getCanvas();
    if (canvas) canvas.style.cursor = props.drawing ? "crosshair" : "";
  }, [props.drawing]);

  const { baseline, result, lines, activeId, view, toggles } = props;
  const after = view === "after" && result !== null;

  useEffect(() => setSelected(null), [result]);
  const selectedStreet = after && selected !== null ? result!.streets.find((s) => s.street_id === selected) : undefined;

  const streetName = useMemo(() => new Map(baseline?.streets.map((s) => [s.street_id, s.name]) ?? []), [baseline]);

  const flowRows: FlowRow[] = useMemo(() => {
    if (!result || !after) return [];
    if (selectedStreet) {
      const here = selectedStreet;
      const isHere = (p: [number, number]) => Math.abs(p[0] - here.lng) < 1e-4 && Math.abs(p[1] - here.lat) < 1e-4;
      const other = (id: number) => streetName.get(id) ?? `Street ${id}`;
      return (here.flows ?? [])
        .filter((f) => f.to) // old scenario files stored residents' areas only
        .map((f) => {
          const into = isHere(f.to);
          return {
            from: f.from,
            to: f.to,
            egp: f.egp,
            between: { into, fromName: into ? other(f.street_id) : here.name, toName: into ? here.name : other(f.street_id) },
          };
        });
    }
    return toggles.flows && zoom <= FLOWS_MAX_ZOOM ? result.flows : [];
  }, [result, after, selectedStreet, toggles.flows, zoom, streetName]);

  const streets: StreetRow[] = useMemo(() => {
    if (!baseline) return [];
    const byId = new Map(result?.streets.map((s) => [s.street_id, s]) ?? []);
    return baseline.streets
      .map((s) => ({ ...s, res: byId.get(s.street_id) }))
      .sort((a, b) => Math.abs(a.res?.dR_pct ?? 0) - Math.abs(b.res?.dR_pct ?? 0)); // changed ones on top
  }, [baseline, result]);

  const drawnStations: DrawnStation[] = useMemo(() => {
    const byKey = new Map(result?.stations.map((s) => [`${s.line_id}:${s.idx}`, s]) ?? []);
    return lines
      .filter((l) => l.visible)
      .flatMap((l) =>
        l.stations.map((s, idx) => ({
          lineId: l.id,
          idx,
          lat: s.lat,
          lng: s.lng,
          color: l.color,
          riders: byKey.get(`${l.id}:${idx}`)?.riders_daily,
          zone: byKey.get(`${l.id}:${idx}`),
          active: l.id === activeId,
        })),
      );
  }, [lines, result, activeId]);

  const zones: ZoneRow[] = useMemo(() => {
    if (!result) return [];
    const color = new Map(lines.filter((l) => l.visible).map((l) => [l.id, l.color]));
    return result.stations
      .filter((s) => s.zone_radius_m > 0 && color.has(s.line_id))
      .map((s) => ({ ...s, color: color.get(s.line_id)! }))
      .sort((a, b) => b.zone_radius_m - a.zone_radius_m); // small circles on top
  }, [result, lines]);

  const microbusHit = useMemo(() => {
    if (!baseline || !result) return null;
    const status = new Map(
      result.microbus_routes.filter((r) => r.status !== "unaffected").map((r) => [r.route_id, r]),
    );
    return {
      type: "FeatureCollection" as const,
      features: baseline.microbus_routes.features
        .filter((f) => status.has(f.properties.route_id))
        .map((f) => ({ ...f, properties: { ...f.properties, ...status.get(f.properties.route_id)! } })),
    };
  }, [baseline, result]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const layers: (Layer | false | null | undefined)[] = [];
    if (baseline) {
      layers.push(
        toggles.allMicrobus &&
          new GeoJsonLayer({
            id: "microbus-all",
            data: baseline.microbus_routes as never,
            getLineColor: [120, 120, 120, 35],
            lineWidthMinPixels: 1,
          }),
        new GeoJsonLayer({
          id: "existing-lines",
          data: baseline.existing_network as never,
          getLineColor: [90, 90, 90, 200],
          lineWidthMinPixels: 3,
        }),
        new ScatterplotLayer({
          id: "existing-stations",
          data: baseline.existing_stations,
          getPosition: (d) => [d.lng, d.lat],
          getRadius: 3.5,
          radiusUnits: "pixels",
          getFillColor: [255, 255, 255],
          getLineColor: [90, 90, 90],
          stroked: true,
          lineWidthMinPixels: 1.5,
          pickable: true,
        }),
      );
    }
    if (result && after) {
      layers.push(
        toggles.microbus &&
          microbusHit &&
          new GeoJsonLayer({
            id: "microbus-hit",
            data: microbusHit as never,
            getLineColor: (f: { properties: { status: string } }) =>
              f.properties.status === "competing" ? [240, 120, 0, 210] : [40, 160, 70, 190],
            lineWidthMinPixels: 2.5,
            pickable: true,
          }),
        toggles.rent &&
          new H3HexagonLayer({
            id: "rent-hexes",
            data: result.hexes,
            getHexagon: (d) => d.h3,
            getFillColor: (d) => sequentialColor(d.rent_pct, Math.max(result.totals.max_rent_pct, 0.5), 150),
            stroked: false,
            extruded: false,
            pickable: true,
          }),
        toggles.footfall &&
          new H3HexagonLayer({
            id: "footfall-cells",
            data: result.footfall_cells,
            getHexagon: (d) => d.h3,
            getFillColor: (d) =>
              sequentialColor(d.footfall_daily, Math.max(...result.footfall_cells.map((c) => c.footfall_daily), 1), 140),
            stroked: false,
            extruded: false,
            pickable: true,
          }),
        toggles.zones &&
          new ScatterplotLayer<ZoneRow>({
            id: "station-zones",
            data: zones,
            getPosition: (d) => [d.lng, d.lat],
            getRadius: (d) => d.zone_radius_m,
            radiusUnits: "meters",
            getFillColor: (d) => hexToRgb(d.color, 38),
            stroked: true,
            getLineColor: (d) => hexToRgb(d.color, 200),
            getLineWidth: 1.5,
            lineWidthUnits: "pixels",
            pickable: true,
          }),
      );
    }
    layers.push(
      new ScatterplotLayer<StreetRow>({
        id: "streets",
        data: streets,
        getPosition: (d) => [d.lng, d.lat],
        getRadius: (d) => streetRadius(d.R0_total),
        radiusUnits: "meters",
        radiusMinPixels: 2,
        radiusMaxPixels: 26,
        getFillColor: (d) => (after ? divergingColor(d.res?.dR_pct ?? 0) : [120, 120, 120, 110]),
        stroked: true,
        getLineColor: (d) => (after && d.res?.risk === "red" ? [120, 0, 0, 255] : [255, 255, 255, 160]),
        getLineWidth: (d) => (after && d.res?.risk === "red" ? 2.5 : 0.5),
        lineWidthUnits: "pixels",
        pickable: true,
        updateTriggers: { getFillColor: [after, result], getLineColor: [after, result], getLineWidth: [after, result] },
      }),
    );
    if (flowRows.length) {
      const maxFlow = Math.max(...flowRows.map((f) => Math.abs(f.egp)), 1);
      const rgba = (c: RGBA, alpha: number): RGBA => [c[0], c[1], c[2], alpha];
      const tint = (d: FlowRow, alpha: number): RGBA => rgba(d.egp < 0 ? LOSS : GAIN, alpha);
      layers.push(
        new ArcLayer<FlowRow>({
          id: "flows",
          data: flowRows,
          getSourcePosition: (d) => d.from,
          getTargetPosition: (d) => d.to,
          // flat on the map: raised arcs seen from straight above bend into loops like flight paths
          getHeight: 0,
          // a clicked street's moves: red at the street losing the money → blue at the one gaining it.
          // Overview: faint where the residents live, solid at the street (blue gain, red loss).
          getSourceColor: (d) => (d.between ? rgba(LOSS, 200) : tint(d, 50)),
          getTargetColor: (d) => (d.between ? rgba(GAIN, 230) : tint(d, 230)),
          getWidth: (d) => 2 + (6 * Math.abs(d.egp)) / maxFlow,
          widthUnits: "pixels",
          pickable: true,
        }),
        // arrowheads point where the spending goes. A clicked street: every move, at the street that
        // gains. Overview: gains only (residents → street); its red lines get none, since that money
        // goes to other shops, not to the residents' end.
        new IconLayer<FlowRow>({
          id: "flow-arrows",
          data: selectedStreet ? flowRows : flowRows.filter((f) => f.egp > 0),
          getPosition: (d) => d.to,
          getIcon: (d) => (d.between?.into ? ARROW_RING : ARROW),
          getAngle: flowAngle,
          billboard: false, // lie on the map so the angle follows map north if the map is rotated
          getSize: (d) => (d.between?.into ? ARROW_RING_PX : 12 + (8 * Math.abs(d.egp)) / maxFlow),
          sizeUnits: "pixels",
          getColor: (d) => (d.between ? rgba(GAIN, 240) : tint(d, 240)),
        }),
      );
    }
    if (selectedStreet) {
      layers.push(
        new ScatterplotLayer<StreetResult>({
          id: "selected-street",
          data: [selectedStreet],
          getPosition: (d) => [d.lng, d.lat],
          getRadius: 13,
          radiusUnits: "pixels",
          filled: false,
          stroked: true,
          getLineColor: [30, 30, 30, 230],
          getLineWidth: 2.5,
          lineWidthUnits: "pixels",
        }),
      );
    }
    layers.push(
      new PathLayer<DraftLine>({
        id: "drawn-lines",
        data: lines.filter((l) => l.visible && l.stations.length >= 2),
        getPath: (l) => l.stations.map((s) => [s.lng, s.lat] as [number, number]),
        getColor: (l) => hexToRgb(l.color, l.id === activeId ? 255 : 190),
        getWidth: (l) => (l.id === activeId ? 6 : 4),
        widthUnits: "pixels",
        capRounded: true,
        jointRounded: true,
        updateTriggers: { getColor: activeId, getWidth: activeId },
      }),
      new ScatterplotLayer<DrawnStation>({
        id: "drawn-stations",
        data: drawnStations,
        getPosition: (d) => [d.lng, d.lat],
        getRadius: (d) => (after && d.riders ? 5 + Math.sqrt(d.riders) / 18 : 6),
        radiusUnits: "pixels",
        getFillColor: (d) => hexToRgb(d.color),
        stroked: true,
        getLineColor: [255, 255, 255],
        lineWidthMinPixels: 2,
        pickable: true,
        updateTriggers: { getRadius: [after, result] },
      }),
    );
    overlay.setProps({ layers: layers.filter(Boolean) as Layer[], getTooltip: tooltip });
  }, [baseline, result, after, toggles, streets, lines, drawnStations, activeId, microbusHit, zones, flowRows, selectedStreet]);

  let hint: JSX.Element | null = null;
  if (selectedStreet)
    hint = (
      <>
        {selectedStreet.dR_egp >= 0 ? (
          <>
            Where <b>{selectedStreet.name}</b>’s extra spending comes from
          </>
        ) : (
          <>
            Where <b>{selectedStreet.name}</b>’s lost spending goes
          </>
        )}
        : each arrow runs from the street that loses the money (<span style={{ color: "rgb(214,64,69)" }}>red</span> end) to
        the one that gains it (<span style={{ color: "rgb(33,113,181)" }}>blue</span> end). Click elsewhere to clear.
      </>
    );
  else if (after && toggles.flows && zoom > FLOWS_MAX_ZOOM && !props.drawing)
    hint = <>Zoomed in: flows are hidden. Click a coloured street to see where its spending moves.</>;
  else if (after && !toggles.flows && !props.drawing)
    hint = <>Click a blue or red street to see where its spending moves.</>;

  return (
    <>
      <div ref={container} className="map" />
      {hint && <div className="map-hint">{hint}</div>}
    </>
  );
}

function row(label: string, value: string) {
  return `<div class="tt-row"><span>${label}</span><b>${value}</b></div>`;
}

function esc(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

function zoneRows(z: StationResult) {
  if (z.zone_radius_m <= 0)
    return row("Likely users", num(z.zone_users)) + `<div class="tt-note">No area of effect: nobody nearby is likely enough to use it</div>`;
  return (
    row("Area of effect", `${(z.zone_radius_m / 1000).toFixed(1)} km radius`) +
    row("Avg chance inside", `${Math.round(100 * z.zone_mean_p)}% of trips`) +
    row("Residents inside", num(z.zone_pop)) +
    row("Likely users", num(z.zone_users))
  );
}

function tooltip(info: PickingInfo) {
  const o = info.object as Record<string, any> | undefined;
  if (!o || !info.layer) return null;
  let html = "";
  switch (info.layer.id) {
    case "streets": {
      const r: StreetResult | undefined = o.res;
      html = `<div class="tt-title">${esc(o.name)}</div>` + row("Baseline spending", egp(o.R0_total) + "/yr");
      if (r) {
        html +=
          row("Change", `${egp(r.dR_egp, true)} (${pct(r.dR_pct)})`) +
          (r.passby_egp ? row("Pass-by from stations", egp(r.passby_egp, true)) : "") +
          row("Rent index", pct(r.rent_pct)) +
          row("Risk", `<span style="color:${RISK_COLOR[r.risk]}">${r.risk}</span>`) +
          (r.local_loser ? `<div class="tt-note">Loses although a new station is close (two-way road)</div>` : "") +
          (r.flows?.length
            ? `<div class="tt-note">Click to see where its ${r.dR_egp >= 0 ? "extra spending comes from" : "lost spending goes"}</div>`
            : "");
      }
      break;
    }
    case "drawn-stations":
      html =
        `<div class="tt-title">Line ${esc(o.lineId)} · station ${o.idx + 1}</div>` +
        (o.riders ? row("Riders / day", num(o.riders)) : "") +
        (o.zone ? zoneRows(o.zone) : "") +
        `<div class="tt-note">Right-click to delete</div>`;
      break;
    case "station-zones":
      html = `<div class="tt-title">Line ${esc(o.line_id)} · station ${o.idx + 1} · area of effect</div>` + zoneRows(o as StationResult);
      break;
    case "existing-stations":
      html = `<div class="tt-title">${esc(o.name)}</div>` + row("Line", esc(o.line)) + row("Residents ≤ 10 min walk", num(o.catchment_pop10));
      break;
    case "microbus-hit":
      html =
        `<div class="tt-title">Microbus ${esc(String(o.properties.route_id))}</div>` +
        row("Status", o.properties.status) +
        row("Trips that would switch", pct(100 * o.properties.share_at_risk, 0).replace("+", ""));
      break;
    case "rent-hexes":
      html = row("Rent index", pct(o.rent_pct)) + row("At opening", pct(o.rent_pct_opening));
      break;
    case "footfall-cells":
      html = row("Station footfall / day", num(o.footfall_daily));
      break;
    case "flows":
      html = o.between
        ? `<div class="tt-title">${esc(o.between.fromName)} → ${esc(o.between.toName)}</div>` +
          row("Spending moved", egp(o.egp) + "/yr") +
          `<div class="tt-note">Shoppers now spend this at ${esc(o.between.toName)} instead of ${esc(o.between.fromName)}</div>`
        : row(o.egp < 0 ? "Street loses from these residents" : "Street gains from these residents", egp(o.egp, true) + "/yr") +
          `<div class="tt-note">${o.egp < 0 ? "They now shop elsewhere instead" : "They now shop here more"}</div>`;
      break;
    default:
      return null;
  }
  return {
    html,
    className: "tooltip",
    style: { backgroundColor: "#fff", color: "#222", padding: "8px 10px", borderRadius: "6px", fontSize: "12px", boxShadow: "0 2px 10px rgba(0,0,0,.2)", maxWidth: "280px" },
  };
}
