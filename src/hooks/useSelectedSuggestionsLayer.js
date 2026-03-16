import { useEffect, useRef } from "react";

const SOURCE_ID = "selected-suggestions-source";
const FILL_ID   = "selected-suggestions-fill";
const LINE_ID   = "selected-suggestions-line";

/**
 * Manages a MapLibre fill+line layer (green) for user-selected suggestion
 * polygons.  Visible at meso and micro zoom; hidden at macro.
 *
 * @param {React.RefObject} mapRef
 * @param {GeoJSON.Feature[]} selectedFeatures – flat array of selected suggestion features
 * @param {string} viewLevel – "macro" | "meso" | "micro"
 */
export function useSelectedSuggestionsLayer(mapRef, selectedFeatures, viewLevel) {
  const addedRef  = useRef(false);
  const latestRef = useRef({ selectedFeatures, viewLevel });
  latestRef.current = { selectedFeatures, viewLevel };

  // ── Effect 1: one-time setup ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const setup = () => {
      if (addedRef.current) return;

      map.addSource(SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

      map.addLayer({
        id:     FILL_ID,
        type:   "fill",
        source: SOURCE_ID,
        layout: { visibility: "none" },
        paint: {
          "fill-color":   "#22c55e",
          "fill-opacity": 0.22,
        },
      });

      map.addLayer({
        id:     LINE_ID,
        type:   "line",
        source: SOURCE_ID,
        layout: { visibility: "none" },
        paint: {
          "line-color":   "#22c55e",
          "line-width":   1.8,
          "line-opacity": 0.85,
        },
      });

      addedRef.current = true;
      syncData(map, latestRef.current.selectedFeatures, latestRef.current.viewLevel);
    };

    if (map.isStyleLoaded()) setup();
    else map.once("load", setup);

    return () => {
      const m = mapRef.current;
      if (!m || !addedRef.current) return;
      try {
        if (m.getLayer(LINE_ID))    m.removeLayer(LINE_ID);
        if (m.getLayer(FILL_ID))    m.removeLayer(FILL_ID);
        if (m.getSource(SOURCE_ID)) m.removeSource(SOURCE_ID);
      } catch { /* map may be gone */ }
      addedRef.current = false;
    };
  }, [mapRef]);

  // ── Effect 2: sync data whenever features or view level change ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !addedRef.current) return;
    syncData(map, selectedFeatures, viewLevel);
  }, [mapRef, selectedFeatures, viewLevel]);
}

function syncData(map, selectedFeatures, viewLevel) {
  const isVisible = (viewLevel === "meso" || viewLevel === "micro") && selectedFeatures.length > 0;
  try {
    map.getSource(SOURCE_ID)?.setData({
      type: "FeatureCollection",
      features: isVisible ? selectedFeatures : [],
    });
    const vis = isVisible ? "visible" : "none";
    map.setLayoutProperty(FILL_ID, "visibility", vis);
    map.setLayoutProperty(LINE_ID, "visibility", vis);
  } catch { /* layers may not be added yet */ }
}