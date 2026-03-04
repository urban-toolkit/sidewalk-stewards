import { useEffect, useState, useCallback } from "react";

let cachedData = null;
let loadingPromise = null;

export function useNetworkData() {
  const [data, setData] = useState(cachedData);

  const load = useCallback(() => {
    const url = `/network.geojson?t=${Date.now()}`;

    loadingPromise = fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch network.geojson: ${res.status}`);
        return res.json();
      })
      .then((fc) => {
        cachedData = fc;
        setData(fc);
        return fc;
      });

    return loadingPromise;
  }, []);

  useEffect(() => {
    if (cachedData) {
      setData(cachedData);
      return;
    }
    load().catch((err) => console.error("Failed to load network data:", err));
  }, [load]);

  const reload = useCallback(() => {
    cachedData = null;
    loadingPromise = null;
    return load();
  }, [load]);

  return { data, reload };
}