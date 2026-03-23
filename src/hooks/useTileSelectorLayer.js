import { useEffect, useRef } from "react";
import { lonLatToTile } from "../utils/tileUtils";

const FILL_SOURCE = "tile-selector-source";
const FILL_LAYER  = "tile-selector-fill";
const LINE_LAYER  = "tile-selector-line";
const EMPTY_FC    = { type: "FeatureCollection", features: [] };

export function useTileSelectorLayer(mapRef, {
  brushActive,
  geojson,
  toggleTile,
  addTilesInBounds,
  setPreviewTiles = () => {},
  dragStart,
  isDragging,
  enabled,
}) {
  const rubberRef  = useRef(null);

  // ── Effect 1: MapLibre source + layers for committed selected tiles ────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !enabled) return;

    let mounted = true; // guard: was setup actually run?

    const setup = () => {
      if (!mounted) return;
      map.addSource(FILL_SOURCE, { type: "geojson", data: EMPTY_FC });
      map.addLayer({
        id: FILL_LAYER, type: "fill", source: FILL_SOURCE,
        paint: { "fill-color": "#4a90d9", "fill-opacity": 0.18 },
      });
      map.addLayer({
        id: LINE_LAYER, type: "line", source: FILL_SOURCE,
        paint: { "line-color": "#4a90d9", "line-width": 2 },
      });
    };

    if (map.isStyleLoaded()) setup();
    else map.once("load", setup);

    return () => {
      mounted = false;
      const m = mapRef.current;
      if (!m) return;
      try {
        if (m.getLayer(FILL_LAYER)) m.removeLayer(FILL_LAYER);
        if (m.getLayer(LINE_LAYER)) m.removeLayer(LINE_LAYER);
        if (m.getSource(FILL_SOURCE)) m.removeSource(FILL_SOURCE);
      } catch { /* map may already be destroyed */ }
    };
  }, [mapRef, enabled]);

  // ── Effect 2: sync committed selection into MapLibre source ───────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !enabled) return;
    try {
      map.getSource(FILL_SOURCE)?.setData(geojson);
    } catch { /* source not ready yet */ }
  }, [mapRef, enabled, geojson]);

  // ── Effect 3: disable / re-enable drag pan + cursor ───────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !enabled) return;
    if (brushActive) {
      map.dragPan.disable();
      map.getCanvas().style.cursor = "crosshair";
    } else {
      map.dragPan.enable();
      map.getCanvas().style.cursor = "";
      setPreviewTiles(new Set());
      if (rubberRef.current) {
        rubberRef.current.remove();
        rubberRef.current = null;
      }
    }
    return () => {
      const m = mapRef.current;
      if (!m) return;
      m.dragPan.enable();
      m.getCanvas().style.cursor = "";
    };
  }, [mapRef, enabled, brushActive, setPreviewTiles]);

  // ── Effect 4: mouse event handlers ───────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !enabled || !brushActive) return;

    const container = map.getContainer();

    const onMouseDown = (e) => {
      dragStart.current  = { lngLat: e.lngLat, point: e.point };
      isDragging.current = false;

      const div = document.createElement("div");
      div.style.cssText = `
        position: absolute;
        border: 2px solid #4a90d9;
        background: rgba(74,144,217,0.1);
        pointer-events: none;
        z-index: 999;
        box-sizing: border-box;
      `;
      container.appendChild(div);
      rubberRef.current = div;
    };

    const onMouseMove = (e) => {
      if (!dragStart.current || !rubberRef.current) return;
      isDragging.current = true;

      const x0 = dragStart.current.point.x;
      const y0 = dragStart.current.point.y;
      const x1 = e.point.x;
      const y1 = e.point.y;

      rubberRef.current.style.left   = `${Math.min(x0, x1)}px`;
      rubberRef.current.style.top    = `${Math.min(y0, y1)}px`;
      rubberRef.current.style.width  = `${Math.abs(x1 - x0)}px`;
      rubberRef.current.style.height = `${Math.abs(y1 - y0)}px`;

      // Live preview tiles inside current rectangle
      const { lng: lng0, lat: lat0 } = dragStart.current.lngLat;
      const { lng: lng1, lat: lat1 } = e.lngLat;
      const west  = Math.min(lng0, lng1), east  = Math.max(lng0, lng1);
      const south = Math.min(lat0, lat1), north = Math.max(lat0, lat1);

      const nw = lonLatToTile(west, north, 18);
      const se = lonLatToTile(east, south, 18);
      const xMin = Math.min(nw.x, se.x), xMax = Math.max(nw.x, se.x);
      const yMin = Math.min(nw.y, se.y), yMax = Math.max(nw.y, se.y);

      const preview = new Set();
      for (let x = xMin; x <= xMax; x++)
        for (let y = yMin; y <= yMax; y++)
          preview.add(`${x}_${y}`);
      setPreviewTiles(preview);
    };

    const onMouseUp = (e) => {
      if (rubberRef.current) {
        rubberRef.current.remove();
        rubberRef.current = null;
      }
      setPreviewTiles(new Set());

      if (!isDragging.current) {
        const { x, y } = lonLatToTile(e.lngLat.lng, e.lngLat.lat, 18);
        toggleTile(`${x}_${y}`);
      } else {
        const { lng: lng0, lat: lat0 } = dragStart.current.lngLat;
        const { lng: lng1, lat: lat1 } = e.lngLat;
        addTilesInBounds({
          west:  Math.min(lng0, lng1), east:  Math.max(lng0, lng1),
          south: Math.min(lat0, lat1), north: Math.max(lat0, lat1),
        });
      }

      dragStart.current  = null;
      isDragging.current = false;
    };

    map.on("mousedown", onMouseDown);
    map.on("mousemove", onMouseMove);
    map.on("mouseup",   onMouseUp);

    return () => {
      map.off("mousedown", onMouseDown);
      map.off("mousemove", onMouseMove);
      map.off("mouseup",   onMouseUp);
      if (rubberRef.current) {
        rubberRef.current.remove();
        rubberRef.current = null;
      }
    };
  }, [mapRef, enabled, brushActive, toggleTile, addTilesInBounds, setPreviewTiles, dragStart, isDragging]);
}