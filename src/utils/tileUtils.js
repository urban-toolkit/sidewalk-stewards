/* Convert lon/lat -> XYZ tile coords (Web Mercator) */
export function lonLatToTile(lon, lat, z) {
  const n = 2 ** z;
  const x = Math.floor(((lon + 180) / 360) * n);

  const latRad = (lat * Math.PI) / 180;
  const y = Math.floor(
    ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n
  );

  return { x, y, z };
}

/* Convert XYZ tile -> lon/lat bounds [W, S, E, N] */
export function tileToLngLatBounds(x, y, z) {
  const n = 2 ** z;

  const west = (x / n) * 360 - 180;
  const east = ((x + 1) / n) * 360 - 180;

  const latRadNorth = Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n)));
  const latRadSouth = Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 1)) / n)));

  const north = (latRadNorth * 180) / Math.PI;
  const south = (latRadSouth * 180) / Math.PI;

  return [west, south, east, north];
}

/* Compute all tiles intersecting bounds at zoom z */
export function tilesForBounds(bounds, z) {
  const nw = lonLatToTile(bounds.west, bounds.north, z);
  const se = lonLatToTile(bounds.east, bounds.south, z);

  const xMin = Math.min(nw.x, se.x);
  const xMax = Math.max(nw.x, se.x);
  const yMin = Math.min(nw.y, se.y);
  const yMax = Math.max(nw.y, se.y);

  const tiles = [];
  for (let x = xMin; x <= xMax; x++) {
    for (let y = yMin; y <= yMax; y++) {
      tiles.push({ z, x, y, id: `${x}_${y}` });
    }
  }

  // Stable order: top-to-bottom, left-to-right
  tiles.sort((a, b) => a.y - b.y || a.x - b.x);
  return tiles;
}

// Get [west, south, east, north] lon/lat for a zoom-18 tile
export function tile18Bounds(x, y) {
  const n    = 2 ** 18;
  const west = (x / n) * 360 - 180;
  const east = ((x + 1) / n) * 360 - 180;
  const latRad = (rad) => (Math.atan(Math.sinh(rad)) * 180) / Math.PI;
  const north  = latRad(Math.PI * (1 - (2 * y) / n));
  const south  = latRad(Math.PI * (1 - (2 * (y + 1)) / n));
  return [west, south, east, north];
}
 
/**
 * Given a lng/lat point, return the zoom-18 tile ID string "x_y".
 */
export function lngLatToTile18(lng, lat) {
  const n      = 2 ** 18;
  const x      = Math.floor(((lng + 180) / 360) * n);
  const latRad = (lat * Math.PI) / 180;
  const y      = Math.floor(
    ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n
  );
  return `${x}_${y}`;
}
 
/**
 * Build a GeoJSON FeatureCollection of tile bounding-box polygons
 * for a Set of zoom-18 tile ID strings.
 */
export function selectedTilesToGeoJSON(tileIds) {
  const features = [...tileIds].map((tid) => {
    const [x, y]       = tid.split("_").map(Number);
    const [w, s, e, n] = tile18Bounds(x, y);
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
 