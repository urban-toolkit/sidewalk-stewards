import { useEffect, useRef, useState, useCallback } from "react";
import maplibregl from "maplibre-gl";

const GOOGLE_KEY = import.meta.env.VITE_GOOGLE_MAPS_KEY;
const MICRO_ZOOM = 18.5;

const RED      = "#ef4444";
const RED_DARK = "#dc2626";

function shortestDelta(from, to) {
  return ((to - from) % 360 + 540) % 360 - 180;
}

function createMarkerElements(heading = 0) {
  const outer = document.createElement("div");
  outer.style.cssText = `width:40px;height:40px;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.3));`;

  const inner = document.createElement("div");
  inner.style.cssText = `
    width:40px;height:40px;
    transform:rotate(${heading}deg);
    transform-origin:center center;
    transition:transform 0.1s ease;
  `;
  inner.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 36 36">
      <path d="M 4.48 14.38 A 14 14 0 0 1 31.52 14.38"
        fill="none" stroke="${RED}" stroke-width="7" stroke-linecap="round" opacity="0.82"/>
      <circle cx="18" cy="18" r="6" fill="${RED}"/>
      <circle cx="18" cy="18" r="3.5" fill="${RED_DARK}" opacity="0.5"/>
    </svg>`;
  outer.appendChild(inner);
  return { outer, inner };
}

// brushActive is passed as a plain boolean; we track it via a ref so the
// click handler never needs to be re-registered when it changes.
export function useStreetView(mapRef, mapZoom, brushActive) {
  const [panel, setPanel] = useState({
    open: false, location: null, heading: 0, loading: false, error: null,
  });

  const zoomRef          = useRef(mapZoom);
  const brushRef         = useRef(brushActive);  // ← gate for click handler
  const markerRef        = useRef(null);
  const innerElRef       = useRef(null);
  const accumulatedAngle = useRef(0);

  // Keep refs in sync with latest prop values each render
  zoomRef.current  = mapZoom;
  brushRef.current = brushActive;

  const removeMarker = useCallback(() => {
    if (markerRef.current) {
      markerRef.current.remove();
      markerRef.current  = null;
      innerElRef.current = null;
    }
    accumulatedAngle.current = 0;
  }, []);

  const closePanel = useCallback(() => {
    removeMarker();
    setPanel({ open: false, location: null, heading: 0, loading: false, error: null });
  }, [removeMarker]);

  const onPanoChange = useCallback(({ heading, lat, lng }) => {
    if (innerElRef.current) {
      const delta = shortestDelta(accumulatedAngle.current, heading);
      accumulatedAngle.current += delta;
      innerElRef.current.style.transform = `rotate(${accumulatedAngle.current}deg)`;
    }
    if (markerRef.current && lat != null && lng != null) {
      markerRef.current.setLngLat([lng, lat]);
    }
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const handleClick = async (e) => {
      // Skip if brush is active or we're not at micro zoom
      if (brushRef.current) return;
      if (zoomRef.current < MICRO_ZOOM) return;

      const { lng, lat } = e.lngLat;
      removeMarker();
      setPanel({ open: true, location: null, heading: 0, loading: true, error: null });

      try {
        const metaRes = await fetch(
          `https://maps.googleapis.com/maps/api/streetview/metadata` +
          `?location=${lat},${lng}&key=${GOOGLE_KEY}`
        );
        const meta = await metaRes.json();

        if (meta.status !== "OK") {
          setPanel({ open: true, location: null, heading: 0, loading: false,
            error: "No street-level imagery found nearby." });
          return;
        }

        const pLat    = meta.location.lat;
        const pLng    = meta.location.lng;
        const heading = meta.heading ?? 0;
        const { outer, inner } = createMarkerElements(heading);

        const marker = new maplibregl.Marker({ element: outer, anchor: "center" })
          .setLngLat([pLng, pLat])
          .addTo(map);

        markerRef.current        = marker;
        innerElRef.current       = inner;
        accumulatedAngle.current = heading;

        setPanel({ open: true, location: { lat: pLat, lng: pLng }, heading, loading: false, error: null });
      } catch (err) {
        setPanel({ open: true, location: null, heading: 0, loading: false, error: err.message });
      }
    };

    const register = () => map.on("click", handleClick);
    if (map.isStyleLoaded()) register();
    else map.once("load", register);

    return () => {
      map.off("click", handleClick);
      removeMarker();
    };
  }, [mapRef, removeMarker]); // ← no brushActive dep — ref handles it without re-registering

  return { panel, closePanel, onPanoChange };
}