import { useState, useEffect, useCallback } from "react";

let cachedSuggestions = null;

function parseSuggestions(fc) {
  const byTile = new Map();
  for (const feature of fc.features) {
    const { tile_id, n_suggestion } = feature.properties;
    if (!byTile.has(tile_id)) byTile.set(tile_id, new Map());
    const byN = byTile.get(tile_id);
    if (!byN.has(n_suggestion)) byN.set(n_suggestion, []);
    byN.get(n_suggestion).push(feature);
  }
  return byTile;
}

/**
 * Loads /polygons.geojson and returns a nested index:
 *   suggestions: Map<tileId, Map<nSuggestion, GeoJSON.Feature[]>>
 *
 * n_suggestion=0  → original polygons
 * n_suggestion>0  → suggestion variant
 *
 * reload() busts the cache and re-fetches — call it after inference completes.
 */
export function useSuggestions() {
  const [suggestions, setSuggestions] = useState(cachedSuggestions);

  const load = useCallback(() => {
    return fetch(`/polygons.geojson?t=${Date.now()}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch polygons.geojson: ${r.status}`);
        return r.json();
      })
      .then((fc) => {
        const parsed = parseSuggestions(fc);
        cachedSuggestions = parsed;
        setSuggestions(parsed);
      })
      .catch((err) => console.error("Failed to load suggestions:", err));
  }, []);

  useEffect(() => {
    if (cachedSuggestions) {
      setSuggestions(cachedSuggestions);
      return;
    }
    load();
  }, [load]);

  const reload = useCallback(() => {
    cachedSuggestions = null;
    return load();
  }, [load]);

  return { suggestions, reload };
}