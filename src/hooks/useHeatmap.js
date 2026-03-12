import { useEffect, useRef, useCallback, useState, useMemo } from "react";
import { tileToLngLatBounds } from "../utils/tileUtils";

const SOURCE_ID = "heatmap-field-source";
const LAYER_ID  = "heatmap-field-layer";

/* ── colour ramp (blue → light → red) ──────────────────────────────── */

function lerpColor(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

const STOPS = [
  { t: 0.0,  rgb: [33,  102, 172] },
  { t: 0.25, rgb: [103, 169, 207] },
  { t: 0.5,  rgb: [253, 219, 199] },
  { t: 0.75, rgb: [239, 138, 98]  },
  { t: 1.0,  rgb: [178, 24,  43]  },
];

function colorForValue(t) {
  if (t <= 0) return STOPS[0].rgb;
  if (t >= 1) return STOPS[STOPS.length - 1].rgb;
  for (let i = 0; i < STOPS.length - 1; i++) {
    if (t >= STOPS[i].t && t <= STOPS[i + 1].t) {
      const local = (t - STOPS[i].t) / (STOPS[i + 1].t - STOPS[i].t);
      return lerpColor(STOPS[i].rgb, STOPS[i + 1].rgb, local);
    }
  }
  return STOPS[STOPS.length - 1].rgb;
}

/* ── Bilinear grid interpolation on a canvas ────────────────────────── */

/**
 * @param {Array}       meta2x2    – tile records to actually draw (may be filtered)
 * @param {string}      sortKey    – attribute name
 * @param {number}      canvasW
 * @param {number}      canvasH
 * @param {number}      opacity    – 0–255, applied to drawn pixels
 * @param {number|null} globalMin  – if provided, overrides the computed min for normalisation
 * @param {number|null} globalMax  – if provided, overrides the computed max for normalisation
 */
function renderBilinear(meta2x2, sortKey, canvasW, canvasH, opacity = 170, globalMin = null, globalMax = null) {
  if (!meta2x2 || meta2x2.length === 0) return null;

  // ── 1. Parse grid coords and values ────────────────────────────────
  const rawValues = meta2x2.map((r) => Number(r[sortKey] ?? 0));

  // Use caller-supplied global range if provided; otherwise derive from this subset
  const min   = globalMin ?? Math.min(...rawValues);
  const max   = globalMax ?? Math.max(...rawValues);
  const range = max - min || 1;

  const parsed = meta2x2.map((r, i) => {
    const [xStr, yStr] = r.tile_id.split("_");
    return { gx: Number(xStr), gy: Number(yStr), v: (rawValues[i] - min) / range };
  });

  // Grid extent
  const gxs  = parsed.map((p) => p.gx);
  const gys  = parsed.map((p) => p.gy);
  const gxMin = Math.min(...gxs);
  const gxMax = Math.max(...gxs);
  const gyMin = Math.min(...gys);
  const gyMax = Math.max(...gys);
  const cols  = gxMax - gxMin + 1;
  const rows  = gyMax - gyMin + 1;

  // ── 2. Fill 2D grid — NaN = missing / filtered-out cell ────────────
  const grid = new Float32Array(rows * cols).fill(NaN);
  for (const p of parsed) {
    const ix = p.gx - gxMin;
    const iy = p.gy - gyMin;
    grid[iy * cols + ix] = p.v;
  }

  // Helper: read grid with clamping (edges repeat border value)
  const gridAt = (ix, iy) => {
    ix = Math.max(0, Math.min(cols - 1, ix));
    iy = Math.max(0, Math.min(rows - 1, iy));
    const v = grid[iy * cols + ix];
    return Number.isNaN(v) ? 0 : v;
  };

  // Whether a cell is present (non-NaN) in the drawn subset
  const cellPresent = (ix, iy) => {
    ix = Math.max(0, Math.min(cols - 1, ix));
    iy = Math.max(0, Math.min(rows - 1, iy));
    return !Number.isNaN(grid[iy * cols + ix]);
  };

  // ── 3. Geographic bounds ────────────────────────────────────────────
  const tileGeoW =
    tileToLngLatBounds(gxMin + 1, gyMin, 18)[0] -
    tileToLngLatBounds(gxMin,     gyMin, 18)[0];
  const tileGeoH =
    tileToLngLatBounds(gxMin, gyMin,     18)[3] -
    tileToLngLatBounds(gxMin, gyMin + 1, 18)[3];

  const [wOrig, , , nOrig] = tileToLngLatBounds(gxMin, gyMin, 18);
  const [,  sOrig, eOrig]  = tileToLngLatBounds(gxMax, gyMax, 18);

  const centreW = wOrig  + tileGeoW * 0.5;
  const centreE = eOrig  - tileGeoW * 0.5;
  const centreN = nOrig  - tileGeoH * 0.5;
  const centreS = sOrig  + tileGeoH * 0.5;

  const west  = centreW - tileGeoW * 0.5;
  const east  = centreE + tileGeoW * 0.5;
  const north = centreN + tileGeoH * 0.5;
  const south = centreS - tileGeoH * 0.5;

  // ── 4. Render bilinear canvas ───────────────────────────────────────
  const canvas = document.createElement("canvas");
  canvas.width  = canvasW;
  canvas.height = canvasH;
  const ctx     = canvas.getContext("2d");
  const imgData = ctx.createImageData(canvasW, canvasH);
  const data    = imgData.data;

  for (let row = 0; row < canvasH; row++) {
    for (let col = 0; col < canvasW; col++) {
      const gxF = ((col + 0.5) / canvasW) * cols - 0.5;
      const gyF = ((row + 0.5) / canvasH) * rows - 0.5;

      // Dominant cell for this pixel (used for alpha masking)
      const cellIx = Math.max(0, Math.min(cols - 1, Math.round(gxF)));
      const cellIy = Math.max(0, Math.min(rows - 1, Math.round(gyF)));

      // If the tile this pixel belongs to was filtered out → transparent
      if (!cellPresent(cellIx, cellIy)) continue;

      // Bilinear blend from the four surrounding nodes
      const ix0 = Math.floor(gxF);
      const iy0 = Math.floor(gyF);
      const ix1 = ix0 + 1;
      const iy1 = iy0 + 1;
      const fx  = gxF - ix0;
      const fy  = gyF - iy0;

      const t =
        gridAt(ix0, iy0) * (1 - fx) * (1 - fy) +
        gridAt(ix1, iy0) * fx       * (1 - fy) +
        gridAt(ix0, iy1) * (1 - fx) * fy +
        gridAt(ix1, iy1) * fx       * fy;

      const [r, g, b] = colorForValue(t);
      const idx = (row * canvasW + col) * 4;
      data[idx]     = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = opacity;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return { dataUrl: canvas.toDataURL(), bounds: [west, south, east, north], min, max };
}

/* ── Hook ───────────────────────────────────────────────────────────── */

/**
 * @param {React.RefObject} mapRef
 * @param {Array|null}      meta2x2   – full z18 metadata
 * @param {string}          sortKey
 * @param {boolean}         visible
 * @param {Set<string>|null} filterIds – z16 tile_ids passing the PCP filter (null = all)
 */
export function useHeatmap(mapRef, meta2x2, sortKey, visible = true, filterIds = null) {
  const layerAdded = useRef(false);
  const [valueRange, setValueRange] = useState({ min: 0, max: 1 });

  // ── Global min/max from the FULL dataset — never changes with filter ─
  const globalRange = useMemo(() => {
    if (!meta2x2 || meta2x2.length === 0) return null;
    const vals = meta2x2.map((r) => Number(r[sortKey] ?? 0));
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, [meta2x2, sortKey]);

  // ── Subset of meta2x2 that passes the filter ────────────────────────
  // PCP filter IDs are z16; each z18 tile's parent = floor(x/4)_floor(y/4)
  const filteredMeta = useMemo(() => {
    if (!meta2x2) return null;
    if (!filterIds) return meta2x2;
    return meta2x2.filter((r) => {
      const [x, y] = r.tile_id.split("_").map(Number);
      const parentId = `${Math.floor(x / 4)}_${Math.floor(y / 4)}`;
      return filterIds.has(parentId);
    });
  }, [meta2x2, filterIds]);

  // ── Toggle visibility ────────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !layerAdded.current) return;
    if (map.getLayer(LAYER_ID)) {
      map.setLayoutProperty(LAYER_ID, "visibility", visible ? "visible" : "none");
    }
  }, [mapRef, visible]);

  const render = useCallback(() => {
    const map = mapRef.current;
    if (!map || !filteredMeta || filteredMeta.length === 0) return;

    const xs     = [...new Set(filteredMeta.map((r) => r.tile_id.split("_")[0]))];
    const ys     = [...new Set(filteredMeta.map((r) => r.tile_id.split("_")[1]))];
    const gridW  = xs.length;
    const gridH  = ys.length;
    const scale  = 8;
    const canvasW = Math.max(64, Math.min(gridW * scale, 1024));
    const canvasH = Math.max(64, Math.min(gridH * scale, 1024));

    const result = renderBilinear(
      filteredMeta,
      sortKey,
      canvasW,
      canvasH,
      170,
      globalRange?.min ?? null,
      globalRange?.max ?? null,
    );
    if (!result) return;

    // Always expose the global range in the legend
    setValueRange({ min: globalRange?.min ?? result.min, max: globalRange?.max ?? result.max });

    const { dataUrl, bounds: geoBounds } = result;
    const [w, s, e, n] = geoBounds;
    const coords = [[w, n], [e, n], [e, s], [w, s]];

    if (!layerAdded.current) {
      const addLayer = () => {
        if (map.getSource(SOURCE_ID)) {
          map.getSource(SOURCE_ID).updateImage({ url: dataUrl, coordinates: coords });
          layerAdded.current = true;
          return;
        }
        map.addSource(SOURCE_ID, { type: "image", url: dataUrl, coordinates: coords });
        map.addLayer({
          id: LAYER_ID,
          type: "raster",
          source: SOURCE_ID,
          paint: { "raster-opacity": 0.65, "raster-fade-duration": 0 },
          layout: { visibility: visible ? "visible" : "none" },
        });
        layerAdded.current = true;
      };

      if (map.isStyleLoaded()) addLayer();
      else map.once("load", addLayer);
      return;
    }

    const src = map.getSource(SOURCE_ID);
    if (src) src.updateImage({ url: dataUrl, coordinates: coords });
  }, [mapRef, filteredMeta, sortKey, visible, globalRange]);

  useEffect(() => { render(); }, [render]);

  // Cleanup
  useEffect(() => {
    return () => {
      const map = mapRef.current;
      if (!map) return;
      try {
        if (map.getLayer(LAYER_ID))  map.removeLayer(LAYER_ID);
        if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);
      } catch { /* map may already be gone */ }
      layerAdded.current = false;
    };
  }, [mapRef]);

  return valueRange;
}