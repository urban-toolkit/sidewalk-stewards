import { useState, useCallback, useRef, useEffect } from "react";
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

/**
 * @param {object} options
 * @param {() => void} [options.onDone] – called after inference completes successfully.
 *   Use this to reload network + polygon data in the parent.
 */
export function useTileSelector({ onDone } = {}) {
  const [brushActive,   setBrushActive]   = useState(false);
  const [selectedTiles, setSelectedTiles] = useState(new Set());
  const [previewTiles,  setPreviewTiles]  = useState(new Set());

  const [inferencePhase,   setInferencePhase]   = useState("idle");
  const [inferenceJobId,   setInferenceJobId]   = useState(null);
  const [inferenceMessage, setInferenceMessage] = useState("");

  // Keep latest onDone in a ref so the polling effect never goes stale
  const onDoneRef = useRef(onDone);
  useEffect(() => { onDoneRef.current = onDone; }, [onDone]);

  // Drag refs — no re-render during mousemove
  const dragStart  = useRef(null);
  const isDragging = useRef(false);

  // ── Drive brushActive from Shift key ──────────────────────────────────────
  useEffect(() => {
    const onKeyDown = (e) => { if (e.key === "Shift" && !e.repeat) setBrushActive(true); };
    const onKeyUp   = (e) => { if (e.key === "Shift") setBrushActive(false); };
    const onBlur    = ()  => setBrushActive(false);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup",   onKeyUp);
    window.addEventListener("blur",    onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup",   onKeyUp);
      window.removeEventListener("blur",    onBlur);
    };
  }, []);

  // ── Poll inference status every 2 s ───────────────────────────────────────
  useEffect(() => {
    if (inferencePhase !== "running" || !inferenceJobId) return;

    console.log(`[inference] job started: ${inferenceJobId}`);

    const id = setInterval(async () => {
      try {
        const res = await fetch(`/api/apply-model/status/${inferenceJobId}`);
        if (!res.ok) {
          console.warn(`[inference] poll returned HTTP ${res.status}`);
          return;
        }
        const { status, message } = await res.json();
        console.log(`[inference] status=${status}${message ? ` | ${message}` : ""}`);

        if (message) setInferenceMessage(message);

        if (status === "done") {
          clearInterval(id);
          console.log("[inference] done ✓ — reloading data and clearing selection");

          // Deselect all tiles immediately
          setSelectedTiles(new Set());

          // Fire the reload callback (reloads network + polygons in App.jsx)
          try { onDoneRef.current?.(); } catch (err) {
            console.warn("[inference] onDone callback threw:", err);
          }

          setInferencePhase("done");
        } else if (status === "error") {
          clearInterval(id);
          console.error("[inference] error:", message);
          setInferencePhase("error");
          setInferenceMessage(message ?? "Unknown error.");
        }
      } catch (err) {
        console.warn("[inference] poll failed:", err);
      }
    }, 2000);

    return () => clearInterval(id);
  }, [inferencePhase, inferenceJobId]);

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
    console.log(`[inference] submitting ${selectedTiles.size} tile(s)…`);
    try {
      const res = await fetch("/api/apply-model", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ tileIds: [...selectedTiles] }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { job_id } = await res.json();
      console.log(`[inference] job_id received: ${job_id}`);
      setInferenceJobId(job_id);
    } catch (err) {
      console.error("[inference] failed to start:", err);
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
    brushActive,
    selectedTiles, toggleTile, addTilesInBounds, clearAll,
    previewTiles, setPreviewTiles,
    dragStart, isDragging,
    runModel, inferencePhase, inferenceJobId, inferenceMessage, dismissInference,
    geojson,
  };
}