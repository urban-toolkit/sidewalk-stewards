"""Generate polygon modification suggestions for sidewalk polygons.

Takes an input polygon file (GeoJSON or SHP), builds a zoom-18 tile grid
from existing zoom-19 tiles, clips polygons to each tile, generates
elongation suggestions, and saves the result as a GeoJSON with tile_id
and n_suggestion attributes.

Usage
-----
python stewards_scripts/generate_suggestions.py \
    --input /path/to/polygons.shp \
    --tiles_dir /path/to/tiles \
    --output ./outputs/polygon_suggestions_zoom18.geojson
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import geopandas as gpd
import pyproj
from shapely.geometry import box as shapely_box
from shapely.ops import transform as shapely_transform
from tqdm.auto import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent / "helper_scripts"))
from polygon_fixing import (
    clip_polygons_to_tile,
    generate_suggestions,
    get_tile_utm_context,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tile coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def num2deg(xtile, ytile, zoom):
    """Convert slippy map tile numbers to lat/lon (top-left corner)."""
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def get_tile_bounds(xtile, ytile, zoom):
    """Get geographic bounding box (west, south, east, north) in EPSG:4326."""
    lat_top, lon_left = num2deg(xtile, ytile, zoom)
    lat_bottom, lon_right = num2deg(xtile + 1, ytile + 1, zoom)
    return lon_left, lat_bottom, lon_right, lat_top


def get_parent(x, y, from_zoom=19, to_zoom=18):
    """Convert tile coordinates from one zoom level to a parent zoom level."""
    shift = from_zoom - to_zoom
    return x >> shift, y >> shift


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate polygon modification suggestions for sidewalk polygons."
    )

    parser.add_argument("--input", default=os.getenv("INPUT_POLYGONS"),
                        help="Input polygon file (GeoJSON or SHP)")
    parser.add_argument("--tiles_dir", default=os.getenv("TILES_DIR"),
                        help="Directory with zoom-19 tile images (*.jpg)")
    parser.add_argument("--output", default="./outputs/polygon_suggestions_zoom18.geojson",
                        help="Output GeoJSON path")

    # Suggestion parameters
    parser.add_argument("--elongation_dist", type=float, default=50.0,
                        help="Extension distance in meters (default: 50)")
    parser.add_argument("--convexity_threshold", type=float, default=0.8,
                        help="Skip polygons below this convexity (default: 0.8)")
    parser.add_argument("--max_elongate", type=int, default=None,
                        help="Max elongation suggestions per tile (default: unlimited)")
    parser.add_argument("--max_compress", type=int, default=0,
                        help="Max compression suggestions per tile (default: 0)")
    parser.add_argument("--max_remove", type=int, default=0,
                        help="Max removal suggestions per tile (default: 0)")
    parser.add_argument("--epsg_utm", type=int, default=32619,
                        help="UTM EPSG code for projection (default: 32619 = Boston)")

    # Road filtering
    parser.add_argument("--filter_across_roads", action="store_true",
                        help="Filter suggestions that cross road polygons")
    parser.add_argument("--road_overlap_min", type=float, default=0.25,
                        help="Min road overlap ratio to start filtering (default: 0.25)")
    parser.add_argument("--road_overlap_max", type=float, default=0.75,
                        help="Max road overlap ratio — above this, keep as crosswalk (default: 0.75)")

    args = parser.parse_args()

    # Validate required paths
    for arg_name in ["input", "tiles_dir"]:
        if getattr(args, arg_name) is None:
            parser.error(f"--{arg_name} is required (set via CLI or .env)")

    # ── Step 1: Load input polygons ──
    print(f"Loading polygons: {args.input}")
    gdf_polys = gpd.read_file(args.input)
    if gdf_polys.crs is None:
        gdf_polys = gdf_polys.set_crs("EPSG:4326")
    elif gdf_polys.crs.to_epsg() != 4326:
        gdf_polys = gdf_polys.to_crs("EPSG:4326")
    print(f"  {len(gdf_polys)} polygons loaded")

    # Separate roads for filtering, then keep only sidewalks
    gdf_roads = None
    if "f_type" in gdf_polys.columns:
        if args.filter_across_roads:
            gdf_roads = gdf_polys[gdf_polys["f_type"] == "road"].copy()
            print(f"  Road polygons for filtering: {len(gdf_roads)}")
        before = len(gdf_polys)
        gdf_polys = gdf_polys[gdf_polys["f_type"] == "sidewalk"].copy()
        print(f"  Filtered to sidewalks: {len(gdf_polys)} (dropped {before - len(gdf_polys)} roads/crosswalks)")

    # ── Step 2: Build zoom-18 tile set from zoom-19 tiles ──
    tiles_dir = Path(args.tiles_dir)
    zoom_19_tiles = [p.stem for p in tiles_dir.glob("*.jpg")]
    print(f"  {len(zoom_19_tiles)} zoom-19 tiles found")

    parent_to_children = defaultdict(list)
    for tid in zoom_19_tiles:
        x, y = map(int, tid.split("_"))
        px, py = get_parent(x, y, from_zoom=19, to_zoom=18)
        parent_to_children[(px, py)].append(tid)

    zoom_18_tiles = sorted(parent_to_children.keys())
    print(f"  {len(zoom_18_tiles)} zoom-18 parent tiles")

    # ── Step 3: Process each zoom-18 tile ──
    # Transformer for UTM → 4326 (reused for all suggestions)
    _utm_to_4326 = pyproj.Transformer.from_crs(
        f"EPSG:{args.epsg_utm}", "EPSG:4326", always_xy=True
    )

    all_rows = []
    n_road_filtered_total = [0]  # mutable for inner scope access

    for px, py in tqdm(zoom_18_tiles, desc="Processing zoom-18 tiles"):
        tile_id_18 = f"{px}_{py}"
        bounds = get_tile_bounds(px, py, zoom=18)

        # Clip polygons to zoom-18 bounds
        tile_polys = clip_polygons_to_tile(gdf_polys, bounds)
        if len(tile_polys) == 0:
            continue

        # Add original polygons (n_suggestion=0)
        for _, row in tile_polys.iterrows():
            all_rows.append({
                "geometry": row.geometry,
                "tile_id": tile_id_18,
                "n_suggestion": 0,
            })

        # Project to UTM and generate suggestions
        tile_polys_proj, tile_box_utm = get_tile_utm_context(
            tile_polys, bounds, epsg_utm=args.epsg_utm
        )

        suggestions = generate_suggestions(
            tile_polys_proj, tile_box_utm,
            elongation_dist=args.elongation_dist,
            max_elongate=args.max_elongate,
            max_compress=args.max_compress,
            max_remove=args.max_remove,
            use_centerline=False,
            convexity_threshold=args.convexity_threshold,
        )

        # Get road union for this tile (if filtering enabled)
        tile_roads_union = None
        if gdf_roads is not None and len(gdf_roads) > 0:
            tile_box = shapely_box(*bounds)
            tile_roads = gdf_roads[gdf_roads.intersects(tile_box)]
            if len(tile_roads) > 0:
                from shapely.ops import unary_union
                valid_roads = [g.buffer(0) if not g.is_valid else g for g in tile_roads.geometry.values]
                tile_roads_union = unary_union(valid_roads)

        # Add suggestion polygons (n_suggestion=1..N)
        n_filtered_roads = 0
        sug_idx = 0
        for sug in suggestions:
            mod_poly_utm = sug["modified_polys"].loc[sug["poly_row_idx"], "geometry"]
            # Get only the elongation (subtract original polygon)
            orig_poly_utm = tile_polys_proj.loc[sug["poly_row_idx"], "geometry"]
            elongation_only = mod_poly_utm.difference(orig_poly_utm)
            # Convert to 4326
            mod_poly_4326 = shapely_transform(_utm_to_4326.transform, elongation_only)

            # Road overlap filter
            if tile_roads_union is not None and not mod_poly_4326.is_empty:
                sug_geom = mod_poly_4326.buffer(0) if not mod_poly_4326.is_valid else mod_poly_4326
                sug_area = sug_geom.area
                if sug_area > 0:
                    road_overlap = sug_geom.intersection(tile_roads_union).area / sug_area
                    if args.road_overlap_min <= road_overlap <= args.road_overlap_max:
                        n_filtered_roads += 1
                        continue

            sug_idx += 1
            all_rows.append({
                "geometry": mod_poly_4326,
                "tile_id": tile_id_18,
                "n_suggestion": sug_idx,
            })

        if n_filtered_roads > 0:
            n_road_filtered_total[0] += n_filtered_roads

    # ── Step 4: Save output ──
    n_orig = sum(1 for r in all_rows if r["n_suggestion"] == 0)
    n_sug = sum(1 for r in all_rows if r["n_suggestion"] > 0)
    print(f"\nTotal polygons: {len(all_rows)} ({n_orig} original, {n_sug} suggestions)")
    if n_road_filtered_total[0] > 0:
        print(f"  Filtered {n_road_filtered_total[0]} suggestions crossing roads")

    gdf_out = gpd.GeoDataFrame(all_rows, crs="EPSG:4326")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(out_path, driver="GeoJSON")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
