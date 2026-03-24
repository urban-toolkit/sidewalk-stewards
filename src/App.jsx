import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import { MapView } from "./components/Map";
import { useMetadata } from "./hooks/useMetadata";
import { useValidMeta } from "./hooks/useValidMeta";
import { useTiles } from "./hooks/useTiles";
import { useTileBorders } from "./hooks/useTileBorders";
import { useSuggestions } from "./hooks/useSuggestions";
import { useSuggestionLayer } from "./hooks/useSuggestionLayer";
import { useSelectedSuggestionsLayer } from "./hooks/useSelectedSuggestionsLayer";
import { useTileSelector } from "./hooks/useTileSelector";
import { useTileSelectorLayer } from "./hooks/useTileSelectorLayer";
import { TileRow } from "./components/TileRow";
import { ParallelCoordinateChart } from "./components/ParallelCoordinateChart";
import { BrushControls } from "./components/BrushControls";
import { tileToLngLatBounds } from "./utils/tileUtils";

import "./App.css";

const VIEW_LABELS    = { macro: "8 × 8 Tiles",  meso: "2 × 2 Tiles", micro: "2 × 2 Tiles" };
const LOADING_LABELS = { macro: "Loading 8 × 8 metadata...", meso: "Loading 2 × 2 metadata...", micro: "Loading 2 × 2 metadata..." };
const LEVEL_BADGES   = { macro: "MACRO", meso: "MESO", micro: "MICRO" };

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

function MicroCard({ tile, networkData, features, size, selected, onToggle }) {
  const imgUrl = `/tiles/${tile.z}/${tile.x}/${tile.y}.jpg`;

  const netLines  = useMemo(() => networkPolylines(tile, networkData, size),
    [tile, networkData, size]);
  const polyPaths = useMemo(() => suggestionPaths(tile, features, size),
    [tile, features, size]);

  return (
    <div
      className={`microSuggestionCard ${selected ? "suggestionSelected" : ""}`}
      onClick={onToggle}
      style={{ cursor: "pointer" }}
    >
      <div className="microSuggestionSquare">
        <img src={imgUrl} alt="" className="microSuggestionImg" loading="lazy" />

        {netLines.length > 0 && (
          <svg viewBox={`0 0 ${size} ${size}`} preserveAspectRatio="none" className="microSuggestionSvg">
            {netLines.map((pts, i) => (
              <polyline key={i} points={pts} fill="none"
                stroke="#e85d04" strokeWidth="1.5" strokeLinecap="round" opacity="0.8" />
            ))}
          </svg>
        )}

        {polyPaths.length > 0 && (
          <svg viewBox={`0 0 ${size} ${size}`} preserveAspectRatio="none" className="microSuggestionSvg">
            {polyPaths.map((d, i) => (
              <path key={i} d={d}
                fill="rgba(34,197,94,0.22)" stroke="#22c55e"
                strokeWidth="1.5" strokeLinejoin="round" fillRule="evenodd" />
            ))}
          </svg>
        )}

        <span className={`suggestionCheckbox ${selected ? "checked" : ""}`}>
          {selected && (
            <svg viewBox="0 0 12 12" width="10" height="10">
              <path d="M2.5 6l2.5 2.5 4.5-5" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </span>
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
  const { suggestions, reload: reloadSuggestions } = useSuggestions();

  // ── Suggestion selection state ───────────────────────────────────────────────
  const [selectedKeys, setSelectedKeys] = useState(new Set());

  const selectedFeatures = useMemo(() => {
    if (selectedKeys.size === 0 || !suggestions) return [];
    const features = [];
    for (const key of selectedKeys) {
      const [tileId, nStr] = key.split(":");
      const n = Number(nStr);
      const tileSugg = suggestions.get(tileId);
      if (!tileSugg) continue;
      const feats = tileSugg.get(n);
      if (feats) features.push(...feats);
    }
    return features;
  }, [selectedKeys, suggestions]);

  const reloadNetworkRef = useRef(null);

  const handleInferenceDone = useCallback(() => {
    reloadNetworkRef.current?.();
    reloadSuggestions();
  }, [reloadSuggestions]);

  // ── Tile selector (brush) — state lives here, layer wired inside render prop ──
  const tileSelector = useTileSelector({ onDone: handleInferenceDone });

  // ── Training state ───────────────────────────────────────────────────────────
  // phase: "idle" | "confirming" | "training" | "done" | "error"
  const [trainingPhase,    setTrainingPhase]    = useState("idle");
  const [trainingJobId,    setTrainingJobId]    = useState(null);
  const [trainingMessage,  setTrainingMessage]  = useState("");
  const [trainingProgress, setTrainingProgress] = useState({ epoch: 0, total: 200 });

  const handleTrainClick = useCallback(() => setTrainingPhase("confirming"), []);

  const handleTrainCancel = useCallback(() => setTrainingPhase("idle"), []);

  const handleTrainDismiss = useCallback(() => {
    setTrainingPhase("idle");
    setTrainingJobId(null);
    setTrainingMessage("");
    setTrainingProgress({ epoch: 0, total: 200 });
  }, []);

  const handleTrainConfirm = useCallback(async () => {
    setTrainingPhase("training");
    setTrainingProgress({ epoch: 0, total: 200 });
    const fc = { type: "FeatureCollection", features: selectedFeatures };
    try {
      const res = await fetch("/api/train", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(fc),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { job_id } = await res.json();
      setTrainingJobId(job_id);
      setSelectedKeys(new Set());
    } catch (err) {
      setTrainingPhase("error");
      setTrainingMessage(err.message);
    }
  }, [selectedFeatures]);

  // ── Poll training status every 3 s ──────────────────────────────────────────
  useEffect(() => {
    if (trainingPhase !== "training" || !trainingJobId) return;
    const id = setInterval(async () => {
      try {
        const res = await fetch(`/api/train/status/${trainingJobId}`);
        if (!res.ok) return;
        const { status, message, epoch, total_epochs } = await res.json();
        if (epoch !== undefined) {
          setTrainingProgress({ epoch, total: total_epochs ?? 200 });
        }
        if (status === "done") {
          clearInterval(id);
          setTrainingPhase("done");
          setTrainingMessage(message ?? "");
        } else if (status === "error") {
          clearInterval(id);
          setTrainingPhase("error");
          setTrainingMessage(message ?? "Unknown error.");
        }
      } catch { /* transient — keep polling */ }
    }, 3000);
    return () => clearInterval(id);
  }, [trainingPhase, trainingJobId]);

  return (
    <div className="page">

      {/* ── Confirmation modal ── */}
      {trainingPhase === "confirming" && (
        <div className="confirmOverlay">
          <div className="confirmCard">
            <div className="confirmTitle">Submit for training?</div>
            <div className="confirmBody">
              {selectedKeys.size} suggestion{selectedKeys.size !== 1 ? "s" : ""} across{" "}
              {new Set([...selectedKeys].map((k) => k.split(":")[0])).size} tile
              {new Set([...selectedKeys].map((k) => k.split(":")[0])).size !== 1 ? "s" : ""} will
              be used to fine-tune the model. Training runs in the background and takes several
              minutes. Selections will be cleared after submission.
            </div>
            <div className="confirmActions">
              <button className="confirmCancelBtn" onClick={handleTrainCancel}>Cancel</button>
              <button className="confirmSubmitBtn" onClick={handleTrainConfirm}>Submit</button>
            </div>
          </div>
        </div>
      )}

      <MapView
        meta2x2={meta2x2}
        sortKey={sortKey}
        filterIds={macroFilterIds}
        brushActive={tileSelector.brushActive}
        selectedTiles={tileSelector.selectedTiles}
        previewTiles={tileSelector.previewTiles}
      >
        {({ bounds, mapZoom, flyToTile, fitToTile, networkData, mapRef, reloadNetwork, dirty, saving, handleSave }) => {
          reloadNetworkRef.current = reloadNetwork;
          
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

          const toggleSuggestion = useCallback((tileId, nSuggestion) => {
            setSelectedKeys((prev) => {
              const key = `${tileId}:${nSuggestion}`;
              const next = new Set(prev);
              if (next.has(key)) next.delete(key);
              else next.add(key);
              return next;
            });
          }, []);

          useTileBorders(mapRef, tiles, focusTile);

          // ── Wire tile selector MapLibre layers ──────────────────────────────
          // Disabled at macro; the hook internally no-ops when enabled=false
          useTileSelectorLayer(mapRef, {
            brushActive:      tileSelector.brushActive,
            geojson:          tileSelector.geojson,
            toggleTile:       tileSelector.toggleTile,
            addTilesInBounds: tileSelector.addTilesInBounds,
            setPreviewTiles:  tileSelector.setPreviewTiles,
            dragStart:        tileSelector.dragStart,
            isDragging:       tileSelector.isDragging,
            enabled:          viewLevel !== "macro",
          });

          const focusTileSuggestions = focusTile
            ? (suggestions?.get(focusTile.id) ?? null)
            : null;

          useSuggestionLayer(mapRef, focusTileSuggestions?.get(0) ?? null, viewLevel);
          useSelectedSuggestionsLayer(mapRef, selectedFeatures, viewLevel);

          const microSuggestions = focusTileSuggestions
            ? [...focusTileSuggestions.entries()]
                .filter(([n]) => n > 0)
                .sort(([a], [b]) => a - b)
            : [];

          const displayTiles = focusTile ? [] :
            viewLevel === 'meso' && suggestions
              ? tiles.filter((t) => {
                  const s = suggestions.get(t.id);
                  return s && [...s.keys()].some((n) => n > 0);
                })
              : tiles;
          const showSuggestions = viewLevel === "meso";
          const thumbSize = viewLevel === "macro" ? 220 : 160;

          const getClickHandler = (tile) => {
            switch (viewLevel) {
              case "macro": return () => flyToTile(tile);
              case "meso":
              case "micro": return () => fitToTile(tile);
              default:      return undefined;
            }
          };

          const pct = trainingProgress.total > 0
            ? Math.round((trainingProgress.epoch / trainingProgress.total) * 100)
            : 0;

          return (
            <>
              {viewLevel !== "macro" && (
                <div style={{
                  position: "absolute",
                  top: 12,
                  left: 12,
                  zIndex: 20,
                  pointerEvents: "none",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  gap: 8,
                }}>
                  {/* Save network — sits above brush controls */}
                  {dirty && (
                    <button
                      className="saveNetworkBtn"
                      onClick={handleSave}
                      disabled={saving}
                      style={{ pointerEvents: "all", position: "static" }}
                    >
                      {saving ? "Saving…" : "Save network"}
                    </button>
                  )}

                  <BrushControls
                    brushActive={tileSelector.brushActive}
                    selectedCount={tileSelector.selectedTiles.size}
                    clearAll={tileSelector.clearAll}
                    runModel={tileSelector.runModel}
                    inferencePhase={tileSelector.inferencePhase}
                    dismissInference={tileSelector.dismissInference}
                    inferenceMessage={tileSelector.inferenceMessage}
                  />
                </div>
              )}

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
                      {viewLevel !== "micro" && (
                        macroFilterIds !== null && viewLevel === "macro"
                          ? `${tiles.length} of ${meta8x8?.length ?? "?"} tiles`
                          : `${tiles.length} tiles`
                      )}
                    </div>

                    {/* Train button */}
                    {trainingPhase === "idle" && selectedKeys.size > 0 && (
                      <button className="trainBtn" onClick={handleTrainClick}>
                        Train model · {selectedKeys.size}
                      </button>
                    )}

                    {/* Training status pill */}
                    {trainingPhase !== "idle" && trainingPhase !== "confirming" && (
                      <div className={`trainingStatus ${trainingPhase}`}>
                        {trainingPhase === "training" && (
                          <>
                            <span className="trainingStatusSpinner" />
                            <span>Training…</span>
                            <div className="trainingProgressBar">
                              <div className="trainingProgressFill" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="trainingProgressPct">{pct}%</span>
                          </>
                        )}
                        {trainingPhase === "done" && (
                          <>
                            Model trained
                            <button className="trainingStatusDismiss" onClick={handleTrainDismiss}>✕</button>
                          </>
                        )}
                        {trainingPhase === "error" && (
                          <>
                            Training failed
                            <button className="trainingStatusDismiss" onClick={handleTrainDismiss}>✕</button>
                          </>
                        )}
                      </div>
                    )}
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
                  <div className="microSuggestionsList">
                    {microSuggestions.length > 0
                      ? microSuggestions.map(([n, features]) => (
                          <MicroCard
                            key={n}
                            tile={focusTile}
                            networkData={networkData}
                            features={features}
                            size={OVERLAY_SIZE}
                            selected={selectedKeys.has(`${focusTile.id}:${n}`)}
                            onToggle={() => toggleSuggestion(focusTile.id, n)}
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
                        selectedKeys={selectedKeys}
                        onToggleSuggestion={toggleSuggestion}
                      />
                    ))}
                  </div>
                )}
              </div>
            </>
          );
        }}
      </MapView>
    </div>
  );
}