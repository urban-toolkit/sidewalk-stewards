import { useState, useEffect, useRef } from "react";
import { useMap } from "../hooks/useMap";
import { useHeatmap } from "../hooks/useHeatmap";
import { useNetworkEditor } from "../hooks/useNetworkEditor";
import { useNetworkData } from "../hooks/useNetworkData";
import { useStreetView } from "../hooks/useStreetView";
import { NetworkEditorMenu } from "./NetworkEditorMenu";
import { StreetViewPanel } from "./StreetViewPanel";
import { tileToLngLatBounds } from "../utils/tileUtils";

function formatValue(v) {
  if (v === undefined || v === null) return "—";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2);
}

// ── Tile selector checkbox overlay ───────────────────────────────────────────
// Renders a small checkbox at the NW corner of every selected tile,
// plus semi-transparent checkbox outlines for preview tiles (under the drag rect).
// Uses map.project() + map "move"/"zoom" events to stay in sync with the canvas.

function TileSelectorOverlay({ mapRef, selectedTiles, previewTiles, brushActive }) {
  const [positions, setPositions] = useState([]);

  useEffect(() => {
    const map = mapRef.current;
    // Combine selected + preview into one projection pass
    const allTiles = new Set([...selectedTiles, ...previewTiles]);

    if (!map || !brushActive || allTiles.size === 0) {
      setPositions([]);
      return;
    }

    const reproject = () => {
      const pts = [];
      for (const tid of allTiles) {
        const [x, y] = tid.split("_").map(Number);
        const [w, , , n] = tileToLngLatBounds(x, y, 18); // NW corner
        const px = map.project([w, n]);
        pts.push({
          tid,
          px: px.x,
          py: px.y,
          isPreview: previewTiles.has(tid) && !selectedTiles.has(tid),
        });
      }
      setPositions(pts);
    };

    reproject();
    map.on("move", reproject);
    map.on("zoom", reproject);
    return () => {
      map.off("move", reproject);
      map.off("zoom", reproject);
    };
  }, [mapRef, selectedTiles, previewTiles, brushActive]);

  if (!brushActive || positions.length === 0) return null;

  return (
    <>
      {positions.map(({ tid, px, py, isPreview }) => (
        <div
          key={tid}
          style={{
            position:        "absolute",
            left:            px + 5,
            top:             py + 5,
            width:           16,
            height:          16,
            borderRadius:    4,
            background:      isPreview ? "rgba(74,144,217,0.25)" : "#4a90d9",
            border:          isPreview ? "1.5px solid #4a90d9" : "1.5px solid #fff",
            boxShadow:       "0 1px 4px rgba(0,0,0,0.2)",
            display:         "flex",
            alignItems:      "center",
            justifyContent:  "center",
            pointerEvents:   "none",
            zIndex:          10,
            transition:      "background 0.1s",
          }}
        >
          {!isPreview && (
            <svg viewBox="0 0 12 12" width="10" height="10">
              <path
                d="M2.5 6l2.5 2.5 4.5-5"
                fill="none" stroke="#fff"
                strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
              />
            </svg>
          )}
        </div>
      ))}
    </>
  );
}

// ── MapView ───────────────────────────────────────────────────────────────────

export function MapView({
  meta2x2,
  sortKey,
  filterIds  = null,
  brushActive = false,
  selectedTiles,
  previewTiles,
  children,
}) {
  const { mapContainerRef, mapRef, bounds, mapZoom, flyToTile, fitToTile } = useMap();
  const [heatmapOn, setHeatmapOn] = useState(true);

  const valueRange = useHeatmap(mapRef, meta2x2, sortKey, heatmapOn && mapZoom < 16, filterIds);

  const { data: networkData, reload: reloadNetwork } = useNetworkData();
  const { contextMenu, setContextMenu, splitEdge, deleteNode, saveNetwork, dirty, saving } =
    useNetworkEditor(mapRef, networkData);

  const { panel: svPanel, closePanel: closeSV, onPanoChange } =
    useStreetView(mapRef, mapZoom, brushActive);

  const handleSave = async () => {
    await saveNetwork();
    reloadNetwork();
  };

  return (
    <>
      <div className="leftPane" style={{ position: "relative" }}>
        <div ref={mapContainerRef} className="map" />

        <TileSelectorOverlay
          mapRef={mapRef}
          selectedTiles={selectedTiles ?? new Set()}
          previewTiles={previewTiles   ?? new Set()}
          brushActive={brushActive}
        />

        <NetworkEditorMenu
          contextMenu={contextMenu}
          setContextMenu={setContextMenu}
          splitEdge={splitEdge}
          deleteNode={deleteNode}
        />

        {dirty && (
          <button className="saveNetworkBtn" onClick={handleSave} disabled={saving}>
            {saving ? "Saving…" : "Save network"}
          </button>
        )}

        {mapZoom < 16 && (
          <div className="mapOverlayControl">
            <label className="toggleLabel">
              <span className="toggleText">Heatmap</span>
              <span
                className={`toggleTrack ${heatmapOn ? "on" : ""}`}
                onClick={() => setHeatmapOn((v) => !v)}
              >
                <span className="toggleThumb" />
              </span>
            </label>
          </div>
        )}

        {heatmapOn && mapZoom < 16 && (
          <div className="mapLegend">
            <div className="legendTitle">{sortKey}</div>
            <div className="legendBar" />
            <div className="legendLabels">
              <span>{formatValue(valueRange.min)}</span>
              <span>{formatValue(valueRange.max)}</span>
            </div>
          </div>
        )}

        <StreetViewPanel panel={svPanel} onClose={closeSV} onPanoChange={onPanoChange} />
      </div>

      {children({ bounds, mapZoom, flyToTile, fitToTile, networkData, mapRef })}
    </>
  );
}