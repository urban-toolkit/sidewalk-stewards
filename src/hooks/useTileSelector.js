import { useState, useCallback, useRef } from "react";
import { lonLatToTile, tileToLngLatBounds } from "../utils/tileUtils";

function buildGeoJSON(tileIds) {
  const features = [...tileIds].map((tid) => {
    const [x, y] = tid.split("_").map(Number);
    const [w, s, e, n] = tileToLngLatBounds(x, y, 18);
    return {
      type: "Feature",
      properties: { tile_id: tid },
      geometry: {
        type: "Polygon",
        coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
      },
    };
  });
  return { type: "FeatureCollection", features };
}

export function useTileSelector() {
  const [brushActive,   setBrushActive]   = useState(false);
  const [selectedTiles, setSelectedTiles] = useState(new Set());
  // Tiles currently under the drag rectangle (live preview, not yet committed)
  const [previewTiles,  setPreviewTiles]  = useState(new Set());

  const [inferencePhase,   setInferencePhase]   = useState("idle");
  const [inferenceJobId,   setInferenceJobId]   = useState(null);
  const [inferenceMessage, setInferenceMessage] = useState("");

  // Drag refs — no re-render during mousemove
  const dragStart  = useRef(null); // { lngLat, point }
  const isDragging = useRef(false);

  const toggleBrush = useCallback(() => setBrushActive((v) => !v), []);

  const toggleTile = useCallback((tileId) => {
    setSelectedTiles((prev) => {
      const next = new Set(prev);
      if (next.has(tileId)) next.delete(tileId);
      else next.add(tileId);
      return next;
    });
  }, []);

  const addTilesInBounds = useCallback(({ west, south, east, north }) => {
    const nw = lonLatToTile(west,  north, 18);
    const se = lonLatToTile(east,  south, 18);
    const xMin = Math.min(nw.x, se.x), xMax = Math.max(nw.x, se.x);
    const yMin = Math.min(nw.y, se.y), yMax = Math.max(nw.y, se.y);
    setSelectedTiles((prev) => {
      const next = new Set(prev);
      for (let x = xMin; x <= xMax; x++)
        for (let y = yMin; y <= yMax; y++)
          next.add(`${x}_${y}`);
      return next;
    });
  }, []);

  const clearAll = useCallback(() => setSelectedTiles(new Set()), []);

  const runModel = useCallback(async () => {
    if (selectedTiles.size === 0) return;
    setInferencePhase("running");
    setInferenceMessage("");
    try {
      const res = await fetch("/api/apply-model", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ tileIds: [...selectedTiles] }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { job_id } = await res.json();
      setInferenceJobId(job_id);
    } catch (err) {
      setInferencePhase("error");
      setInferenceMessage(err.message);
    }
  }, [selectedTiles]);

  const dismissInference = useCallback(() => {
    setInferencePhase("idle");
    setInferenceJobId(null);
    setInferenceMessage("");
  }, []);

  const geojson = buildGeoJSON(selectedTiles);

  return {
    brushActive, toggleBrush,
    selectedTiles, toggleTile, addTilesInBounds, clearAll,
    previewTiles, setPreviewTiles,
    dragStart, isDragging,
    runModel, inferencePhase, inferenceJobId, inferenceMessage, dismissInference,
    geojson,
  };
}