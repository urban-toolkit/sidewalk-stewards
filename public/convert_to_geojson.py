#!/usr/bin/env python3
"""
convert_network.py

Reads the original shapefile (public/original/original_network.*) and writes
a single GeoJSON FeatureCollection to public/network.geojson.

This is the same format that useNetworkEditor.js expects:
each feature is a LineString with coordinate arrays.

Usage:
    python scripts/convert_network.py

Requirements:
    pip install geopandas
"""

import json
import sys
from pathlib import Path

try:
    import geopandas as gpd
except ImportError:
    print("Error: geopandas is required.  Install it with:")
    print("  pip install geopandas")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
SHP_PATH = ROOT / "public" / "original" / "original_network.shp"
OUT_PATH = ROOT / "public" / "network.geojson"


def main():
    if not SHP_PATH.exists():
        print(f"Shapefile not found: {SHP_PATH}")
        sys.exit(1)

    print(f"Reading shapefile: {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH)

    if gdf.crs and not gdf.crs.equals("EPSG:4326"):
        print(f"Reprojecting from {gdf.crs} -> EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        props = {k: v for k, v in row.items() if k != "geometry"}
        props = {k: (v.item() if hasattr(v, "item") else v) for k, v in props.items()}

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": geom.__geo_interface__,
            }
        )

    fc = {"type": "FeatureCollection", "features": features}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(fc, f)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {len(features)} features → {OUT_PATH}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()