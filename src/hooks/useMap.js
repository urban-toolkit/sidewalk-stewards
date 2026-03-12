import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { tileToLngLatBounds } from "../utils/tileUtils";

const MAP_STYLE = {
  version: 8,
  sources: {
    carto: {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© <a href='https://carto.com/'>CARTO</a> © <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>",
    },
  },
  layers: [{ id: "carto", type: "raster", source: "carto" }],
};

export function useMap() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);

  const [bounds, setBounds] = useState(null);
  const [mapZoom, setMapZoom] = useState(12);

  useEffect(() => {
    if (mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAP_STYLE,
      center: [-71.06, 42.3], // Dorchester / Boston area
      zoom: 13,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    map.on("error", (e) => {
      console.error("MAP ERROR:", e?.error || e);
    });

    const syncState = () => {
      const b = map.getBounds();
      setBounds({
        west: b.getWest(),
        south: b.getSouth(),
        east: b.getEast(),
        north: b.getNorth(),
      });
      setMapZoom(map.getZoom());
    };

    map.on("load", () => {
      map.addSource("myOrthoTiles", {
        type: "raster",
        tiles: ["/tiles/{z}/{x}/{y}.jpg"],
        tileSize: 256,
        minzoom: 16,
        maxzoom: 19,
      });

      map.addLayer({
        id: "myOrthoLayer",
        type: "raster",
        source: "myOrthoTiles",
        minzoom: 16,
        paint: { "raster-opacity": 1.0 },
      });

      syncState();
    });

    map.on("moveend", syncState);
    map.on("zoomend", syncState);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  /**
   * Fly to an 8×8 tile (macro → meso transition).
   * Caps zoom at 16.99 so we land in meso level.
   */
  const flyToTile = (tile) => {
    const map = mapRef.current;
    if (!map) return;

    const [w, s, e, n] = tileToLngLatBounds(tile.x, tile.y, tile.z);
    map.fitBounds([[w, s], [e, n]], {
      padding: 10,
      duration: 600,
      maxZoom: 16.99,
    });
  };

  /**
   * Fit the map exactly to a 2×2 tile (meso → micro / micro → micro).
   * No maxZoom cap — lets the map zoom in fully to the tile extent.
   */
  const fitToTile = (tile) => {
    const map = mapRef.current;
    if (!map) return;

    const [w, s, e, n] = tileToLngLatBounds(tile.x, tile.y, tile.z);
    map.fitBounds([[w, s], [e, n]], {
      padding: 10,
      duration: 600
    });
  };

  return { mapContainerRef, mapRef, bounds, mapZoom, flyToTile, fitToTile };
}