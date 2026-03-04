import { useState } from "react";
import { useMap } from "../hooks/useMap";
import { useHeatmap } from "../hooks/useHeatmap";
import { useNetworkEditor } from "../hooks/useNetworkEditor";
import { useNetworkData } from "../hooks/useNetworkData";
import { NetworkEditorMenu } from "./NetworkEditorMenu";

function formatValue(v) {
  if (v === undefined || v === null) return "—";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2);
}

export function MapView({ meta2x2, sortKey, children }) {
  const { mapContainerRef, mapRef, bounds, mapZoom, flyToTile, fitToTile } = useMap();
  const [heatmapOn, setHeatmapOn] = useState(true);

  const valueRange  = useHeatmap(mapRef, meta2x2, sortKey, heatmapOn);
  const { data: networkData, reload: reloadNetwork } = useNetworkData();
  const { contextMenu, setContextMenu, splitEdge, deleteNode, saveNetwork, dirty, saving } =
    useNetworkEditor(mapRef, networkData);

  const handleSave = async () => {
    await saveNetwork();
    reloadNetwork();
  };

  return (
    <>
      {/* ── Map pane ───────────────────────────────────────────────── */}
      <div className="leftPane" style={{ position: "relative" }}>
        <div ref={mapContainerRef} className="map" />

        {/* Network editor context menu */}
        <NetworkEditorMenu
          contextMenu={contextMenu}
          setContextMenu={setContextMenu}
          splitEdge={splitEdge}
          deleteNode={deleteNode}
        />

        {/* Save network button */}
        {dirty && (
          <button
            className="saveNetworkBtn"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving…" : "Save network"}
          </button>
        )}

        {/* Heatmap toggle */}
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

        {/* Colour scale legend */}
        {heatmapOn && (
          <div className="mapLegend">
            <div className="legendTitle">{sortKey}</div>
            <div className="legendBar" />
            <div className="legendLabels">
              <span>{formatValue(valueRange.min)}</span>
              <span>{formatValue(valueRange.max)}</span>
            </div>
          </div>
        )}
      </div>

      {/* ── Right pane (render prop) ────────────────────────────────── */}
      {children({ bounds, mapZoom, flyToTile, fitToTile, networkData, mapRef })}
    </>
  );
}