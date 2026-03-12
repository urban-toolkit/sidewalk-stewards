import { useState } from "react";
import { MapView } from "./components/Map";
import { useMetadata } from "./hooks/useMetadata";
import { useValidMeta } from "./hooks/useValidMeta";
import { useTiles } from "./hooks/useTiles";
import { useTileBorders } from "./hooks/useTileBorders";
import { TileRow } from "./components/TileRow";
import { ParallelCoordinateChart } from "./components/ParallelCoordinateChart";
import { tileToLngLatBounds } from "./utils/tileUtils";
import "./App.css";

const VIEW_LABELS = {
  macro: "8 × 8 Tiles",
  meso:  "2 × 2 Tiles",
  micro: "2 × 2 Tiles",
};

const LOADING_LABELS = {
  macro: "Loading 8 × 8 metadata...",
  meso:  "Loading 2 × 2 metadata...",
  micro: "Loading 2 × 2 metadata...",
};

const LEVEL_BADGES = {
  macro: "MACRO",
  meso:  "MESO",
  micro: "MICRO",
};

function dominantTile(tiles, bounds) {
  if (!tiles.length || !bounds) return tiles[0] ?? null;

  let best     = tiles[0];
  let bestArea = -1;

  for (const t of tiles) {
    const [tw, ts, te, tn] = tileToLngLatBounds(t.x, t.y, t.z);
    const iw   = Math.max(tw, bounds.west);
    const ie   = Math.min(te, bounds.east);
    const is_  = Math.max(ts, bounds.south);
    const in_  = Math.min(tn, bounds.north);
    const area = Math.max(0, ie - iw) * Math.max(0, in_ - is_);
    if (area > bestArea) { bestArea = area; best = t; }
  }

  return best;
}

export default function App() {
  const [sortKey,        setSortKey]        = useState("n_uncertain");
  const [macroFilterIds, setMacroFilterIds] = useState(null);
  const { meta8x8, meta2x2 } = useMetadata();

  // Probe all z16 images once on load; PCP and tile list both use only valid tiles
  const { validMeta8x8, validating } = useValidMeta(meta8x8);

  return (
    <div className="page">
      <MapView meta2x2={meta2x2} sortKey={sortKey} filterIds={macroFilterIds}>
        {({ bounds, mapZoom, flyToTile, fitToTile, networkData, mapRef }) => {
          const { tiles, viewportTileIds, activeMeta, activeMetaById, viewLevel } = useTiles({
            bounds,
            mapZoom,
            meta8x8: validMeta8x8,  // only tiles with real images
            meta2x2,
            sortKey,
            filteredIds: macroFilterIds,
          });

          const focusTile =
            viewLevel === "micro" && tiles.length > 0
              ? dominantTile(tiles, bounds)
              : null;

          useTileBorders(mapRef, tiles, focusTile);

          const displayTiles = focusTile ? [] : tiles;

          const getClickHandler = (tile) => {
            switch (viewLevel) {
              case "macro": return () => flyToTile(tile);
              case "meso":
              case "micro": return () => fitToTile(tile);
              default:      return undefined;
            }
          };

          const showSuggestions = viewLevel === "meso";

          // Thumb size: macro uses cards in a 3-col grid so smaller;
          // meso/micro stay at 160px
          const thumbSize = viewLevel === "macro" ? 220 : 160;

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

              {/* ── Parallel Coordinate Chart (macro only) ── */}
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

              {/* ── Tile content ── */}
              {viewLevel === "micro" && focusTile ? (
                <div className="microSuggestionsGrid">
                  {Array.from({ length: 20 }).map((_, i) => (
                    <div key={i} className="suggestionCard">
                      Suggestion {i + 1}
                    </div>
                  ))}
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