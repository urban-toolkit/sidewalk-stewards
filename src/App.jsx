import { useState, useMemo } from "react";
import { MapView } from "./components/Map";
import { useMetadata } from "./hooks/useMetadata";
import { useValidMeta } from "./hooks/useValidMeta";
import { useTiles } from "./hooks/useTiles";
import { useTileBorders } from "./hooks/useTileBorders";
import { useSuggestions } from "./hooks/useSuggestions";
import { useSuggestionLayer } from "./hooks/useSuggestionLayer";
import { TileRow } from "./components/TileRow";
import { ParallelCoordinateChart } from "./components/ParallelCoordinateChart";
import { tileToLngLatBounds } from "./utils/tileUtils";
import "./App.css";

const VIEW_LABELS  = { macro: "8 × 8 Tiles",  meso: "2 × 2 Tiles", micro: "2 × 2 Tiles" };
const LOADING_LABELS = { macro: "Loading 8 × 8 metadata...", meso: "Loading 2 × 2 metadata...", micro: "Loading 2 × 2 metadata..." };
const LEVEL_BADGES = { macro: "MACRO", meso: "MESO", micro: "MICRO" };

// Logical coordinate space for SVG viewBox — does not affect rendered pixel size
const OVERLAY_SIZE = 256;

// ── Geo → SVG point helpers ───────────────────────────────────────────────────

function geoToSVGPoints(ring, west, north, geoW, geoH, size) {
  return ring.map(([lon, lat]) =>
    `${(((lon - west) / geoW) * size).toFixed(1)},${(((north - lat) / geoH) * size).toFixed(1)}`
  );
}

function networkPolylines(tile, networkData, size) {
  if (!networkData?.features) return [];
  const [west, south, east, north] = tileToLngLatBounds(tile.x, tile.y, tile.z);
  const geoW = east - west;
  const geoH = north - south;
  const lines = [];
  for (const f of networkData.features) {
    const segs = f.geometry.type === "LineString"
      ? [f.geometry.coordinates]
      : f.geometry.coordinates;
    for (const ring of segs) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const [lon, lat] of ring) {
        if (lon < minX) minX = lon; if (lon > maxX) maxX = lon;
        if (lat < minY) minY = lat; if (lat > maxY) maxY = lat;
      }
      if (maxX < west || minX > east || maxY < south || minY > north) continue;
      lines.push(geoToSVGPoints(ring, west, north, geoW, geoH, size).join(" "));
    }
  }
  return lines;
}

function suggestionPaths(tile, features, size) {
  if (!features?.length) return [];
  const [west, south, east, north] = tileToLngLatBounds(tile.x, tile.y, tile.z);
  const geoW = east - west;
  const geoH = north - south;
  const toSeg = (ring) => {
    const pts = geoToSVGPoints(ring, west, north, geoW, geoH, size);
    return `M ${pts.join(" L ")} Z`;
  };
  const paths = [];
  for (const f of features) {
    if (f.geometry.type === "Polygon") {
      paths.push(f.geometry.coordinates.map(toSeg).join(" "));
    } else if (f.geometry.type === "MultiPolygon") {
      for (const poly of f.geometry.coordinates) paths.push(poly.map(toSeg).join(" "));
    }
  }
  return paths;
}

// ── Micro suggestion card ─────────────────────────────────────────────────────

function MicroCard({ tile, networkData, features, size }) {
  const imgUrl = `/tiles/${tile.z}/${tile.x}/${tile.y}.jpg`;

  const netLines  = useMemo(() => networkPolylines(tile, networkData, size),
    [tile, networkData, size]);
  const polyPaths = useMemo(() => suggestionPaths(tile, features, size),
    [tile, features, size]);

  return (
    <div className="microSuggestionCard">
      <div className="microSuggestionSquare">
        <img src={imgUrl} alt="" className="microSuggestionImg" loading="lazy" />

        {/* Network lines */}
        {netLines.length > 0 && (
          <svg
            viewBox={`0 0 ${size} ${size}`}
            preserveAspectRatio="none"
            className="microSuggestionSvg"
          >
            {netLines.map((pts, i) => (
              <polyline key={i} points={pts} fill="none"
                stroke="#e85d04" strokeWidth="1.5" strokeLinecap="round" opacity="0.8" />
            ))}
          </svg>
        )}

        {/* Suggestion polygons */}
        {polyPaths.length > 0 && (
          <svg
            viewBox={`0 0 ${size} ${size}`}
            preserveAspectRatio="none"
            className="microSuggestionSvg"
          >
            {polyPaths.map((d, i) => (
              <path key={i} d={d}
                fill="rgba(59,130,246,0.22)" stroke="#3b82f6"
                strokeWidth="1.5" strokeLinejoin="round" fillRule="evenodd" />
            ))}
          </svg>
        )}
      </div>
    </div>
  );
}

// ── dominantTile ──────────────────────────────────────────────────────────────

function dominantTile(tiles, bounds) {
  if (!tiles.length || !bounds) return tiles[0] ?? null;
  let best = tiles[0], bestArea = -1;
  for (const t of tiles) {
    const [tw, ts, te, tn] = tileToLngLatBounds(t.x, t.y, t.z);
    const area = Math.max(0, Math.min(te, bounds.east) - Math.max(tw, bounds.west))
               * Math.max(0, Math.min(tn, bounds.north) - Math.max(ts, bounds.south));
    if (area > bestArea) { bestArea = area; best = t; }
  }
  return best;
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [sortKey,        setSortKey]        = useState("n_uncertain");
  const [macroFilterIds, setMacroFilterIds] = useState(null);
  const { meta8x8, meta2x2 } = useMetadata();
  const { validMeta8x8, validating } = useValidMeta(meta8x8);
  const { suggestions } = useSuggestions();

  return (
    <div className="page">
      <MapView meta2x2={meta2x2} sortKey={sortKey} filterIds={macroFilterIds}>
        {({ bounds, mapZoom, flyToTile, fitToTile, networkData, mapRef }) => {
          const { tiles, viewportTileIds, activeMeta, activeMetaById, viewLevel } = useTiles({
            bounds, mapZoom,
            meta8x8: validMeta8x8,
            meta2x2, sortKey,
            filteredIds: macroFilterIds,
          });

          const focusTile =
            viewLevel === "micro" && tiles.length > 0
              ? dominantTile(tiles, bounds)
              : null;

          useTileBorders(mapRef, tiles, focusTile);

          const focusTileSuggestions = focusTile
            ? (suggestions?.get(focusTile.id) ?? null)
            : null;

          // n_suggestion=0 → shown as polygon overlay on the map (micro only)
          useSuggestionLayer(mapRef, focusTileSuggestions?.get(0) ?? null, viewLevel);

          // n_suggestion > 0 → shown in right-panel suggestion cards
          const microSuggestions = focusTileSuggestions
            ? [...focusTileSuggestions.entries()]
                .filter(([n]) => n > 0)
                .sort(([a], [b]) => a - b)
            : [];

          const displayTiles  = focusTile ? [] : tiles;
          const showSuggestions = viewLevel === "meso";
          const thumbSize     = viewLevel === "macro" ? 220 : 160;

          const getClickHandler = (tile) => {
            switch (viewLevel) {
              case "macro": return () => flyToTile(tile);
              case "meso":
              case "micro": return () => fitToTile(tile);
              default:      return undefined;
            }
          };

          return (
            <div className={`rightPane ${viewLevel}`}>
              {/* ── Header ── */}
              <div className="header">
                <div>
                  <div className="title">
                    {VIEW_LABELS[viewLevel]}
                    <span className="levelBadge">{LEVEL_BADGES[viewLevel]}</span>
                  </div>
                  <div className="sub">
                    {!activeMeta
                      ? LOADING_LABELS[viewLevel]
                      : viewLevel === "micro" && focusTile
                      ? `Tile ${focusTile.id} · Zoom ${mapZoom.toFixed(2)}`
                      : bounds
                      ? `Zoom ${mapZoom.toFixed(2)} · W ${bounds.west.toFixed(4)} · S ${bounds.south.toFixed(4)} · E ${bounds.east.toFixed(4)} · N ${bounds.north.toFixed(4)}`
                      : "Waiting for map..."}
                  </div>
                </div>
                <div className="controls">
                  <div className="count">
                    {viewLevel === "micro"
                      ? `1 of ${tiles.length} tiles`
                      : macroFilterIds !== null && viewLevel === "macro"
                      ? `${tiles.length} of ${meta8x8?.length ?? "?"} tiles`
                      : `${tiles.length} tiles`}
                  </div>
                </div>
              </div>

              {/* ── PCP (macro only) ── */}
              {viewLevel === "macro" && (
                validating
                  ? <div className="pcpWrapper" style={{ padding: "12px", fontSize: 11, color: "#888" }}>Checking tiles…</div>
                  : validMeta8x8 && (
                    <ParallelCoordinateChart
                      data={validMeta8x8}
                      visibleIds={viewportTileIds}
                      sortKey={sortKey}
                      onSortChange={setSortKey}
                      onFilterChange={setMacroFilterIds}
                    />
                  )
              )}

              {/* ── Content ── */}
              {viewLevel === "micro" && focusTile ? (
                /* 2-col scrollable grid — 4 visible at a time, more on scroll */
                <div className="microSuggestionsList">
                  {microSuggestions.length > 0
                    ? microSuggestions.map(([n, features]) => (
                        <MicroCard
                          key={n}
                          tile={focusTile}
                          networkData={networkData}
                          features={features}
                          size={OVERLAY_SIZE}
                        />
                      ))
                    : (
                        <div className="microNoSuggestions">
                          No suggestions for this area
                        </div>
                      )}
                </div>
              ) : (
                <div className={viewLevel === "macro" ? "macroGrid" : "list"}>
                  {displayTiles.map((t) => (
                    <TileRow
                      key={`${t.z}_${t.id}`}
                      tile={t}
                      meta={activeMetaById?.get(t.id)}
                      sortKey={sortKey}
                      onClick={getClickHandler(t)}
                      showSuggestions={showSuggestions}
                      networkData={networkData}
                      thumbSize={thumbSize}
                      tileSuggestions={suggestions?.get(t.id) ?? null}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        }}
      </MapView>
    </div>
  );
}