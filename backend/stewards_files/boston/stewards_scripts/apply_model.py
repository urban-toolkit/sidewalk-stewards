"""Apply a trained sidewalk fix model and produce global network + polygon GeoJSON.

Takes zoom-18 tile IDs (or a GeoJSON to derive them from), loads a trained
model, runs inference on zoom-19 children, re-polygonizes, runs Tile2Net
topology to produce a sidewalk network. Then merges the new polygons and
network segments into the original global files (replacing the input tiles'
old data).

Usage
-----
python apply_model.py \
    --tile_ids 79325_97025 79322_97010 79303_97006 \
    --model_path ./outputs/suggestion_model.pt \
    --tiles_dir .../tiles \
    --t2n_dir .../masks_tile2net_polygons \
    --conf_dir .../masks_confidence \
    --original_polygons .../polygons_global.geojson \
    --original_network .../network_global.geojson \
    --output_polygons ./outputs/polygons.geojson \
    --output_network ./outputs/network.geojson \
    --head fix

Or derive tile IDs from a GeoJSON:

python apply_model.py \
    --geojson ./suggestion_sample.geojson \
    --model_path ./outputs/suggestion_model.pt \
    ...
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point, box
from shapely.ops import nearest_points, unary_union
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent / "helper_scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from polygon_fixing import get_tile_bounds
from tile2net_training_utils import ResidualFixNet
from train_from_suggestions import (
    mask_to_polygons,
    run_inference,
)

# Tile2Net topology imports (from pip install -e .)
from tile2net.raster.project import Project
from tile2net.raster.raster import Raster
from tile2net.raster.pednet import PedNet


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def zoom18_to_zoom19(tile_ids_18):
    """Expand zoom-18 tile IDs to their 4 zoom-19 children."""
    tile_ids_19 = []
    for tid in tile_ids_18:
        x, y = map(int, tid.split("_"))
        for dx in range(2):
            for dy in range(2):
                tile_ids_19.append(f"{x * 2 + dx}_{y * 2 + dy}")
    return tile_ids_19


def filter_existing_tiles(tile_ids, tiles_dir, t2n_dir, conf_dir):
    """Keep only tile IDs where RGB, T2N, and confidence files exist."""
    tiles_dir = Path(tiles_dir)
    t2n_dir = Path(t2n_dir)
    conf_dir = Path(conf_dir)

    valid = []
    for tid in tile_ids:
        if ((tiles_dir / f"{tid}.jpg").exists()
                and (t2n_dir / f"{tid}.png").exists()
                and (conf_dir / f"{tid}.png").exists()):
            valid.append(tid)
    return valid


def compute_bbox_from_tiles(tile_ids_18):
    """Compute lon/lat bounding box from zoom-18 tile IDs."""
    all_w, all_s, all_e, all_n = [], [], [], []
    for tid in tile_ids_18:
        x, y = map(int, tid.split("_"))
        w, s, e, n = get_tile_bounds(x, y, zoom=18)
        all_w.append(w)
        all_s.append(s)
        all_e.append(e)
        all_n.append(n)
    return (min(all_w), min(all_s), max(all_e), max(all_n))


def polygonize_predictions(predictions):
    """Convert predicted masks to GeoDataFrame with zoom-18 tile_id."""
    all_polys = []
    all_tile_ids = []

    for tid, mask in tqdm(predictions.items(), desc="Re-polygonizing"):
        if mask.max() == 0:
            continue
        xtile, ytile = map(int, tid.split("_"))
        polys = mask_to_polygons(mask, xtile, ytile)
        tile_id_18 = f"{xtile // 2}_{ytile // 2}"
        for p in polys:
            all_polys.append(p)
            all_tile_ids.append(tile_id_18)

    if not all_polys:
        return gpd.GeoDataFrame(columns=["tile_id", "geometry"], crs="EPSG:4326")

    return gpd.GeoDataFrame(
        {"tile_id": all_tile_ids},
        geometry=all_polys,
        crs="EPSG:4326",
    )


def run_topology(gdf_polygons, bbox):
    """Run Tile2Net topology engine on polygons to produce network lines.

    Parameters
    ----------
    gdf_polygons : GeoDataFrame
        Polygon geometries with CRS EPSG:4326.
    bbox : tuple
        (min_lon, min_lat, max_lon, max_lat).

    Returns
    -------
    GeoDataFrame with network line geometries.
    """
    gdf = gdf_polygons.copy()
    if "f_type" not in gdf.columns:
        gdf["f_type"] = "sidewalk"

    print("\nCreating raster + project for topology...")
    raster = Raster(
        location=list(bbox),
        name="polygon_injection",
        zoom=19,
        base_tilesize=256,
    )

    project = Project("polygon_network_output", "outputs", raster)

    print("Running topology engine...")
    pednet = PedNet(gdf, project)
    pednet.convert_whole_poly2line()

    network = pednet.complete_net
    print(f"  {len(network)} network edges produced")
    return network


def _tile_union(tile_ids_18):
    """Build a single geometry covering all zoom-18 tile bounding boxes."""
    tile_boxes = []
    for tid in tile_ids_18:
        x, y = map(int, str(tid).split("_"))
        w, s, e, n = get_tile_bounds(x, y, zoom=18)
        tile_boxes.append(box(w, s, e, n))
    return unary_union(tile_boxes)


def remove_tiles_from_polygons(gdf_polys, tile_ids_18):
    """Remove polygons by tile_id; fall back to spatial if column missing."""
    tile_set = set(str(t) for t in tile_ids_18)
    if "tile_id" in gdf_polys.columns:
        mask = gdf_polys["tile_id"].astype(str).isin(tile_set)
        removed = mask.sum()
        print(f"  Removing {removed} old polygons from {len(tile_set)} tiles (by tile_id)")
        return gdf_polys[~mask].copy()
    else:
        tile_geom = _tile_union(tile_ids_18)
        mask = gdf_polys.intersects(tile_geom)
        removed = mask.sum()
        print(f"  Removing {removed} old polygons from {len(tile_set)} tile areas (spatial)")
        return gdf_polys[~mask].copy()


def remove_tiles_from_network(gdf_net, tile_ids_18):
    """Remove network segments that fall within the given zoom-18 tiles."""
    tile_boxes = []
    for tid in tile_ids_18:
        x, y = map(int, str(tid).split("_"))
        w, s, e, n = get_tile_bounds(x, y, zoom=18)
        tile_boxes.append(box(w, s, e, n))
    tile_union = unary_union(tile_boxes)

    mask = gdf_net.intersects(tile_union)
    removed = mask.sum()
    print(f"  Removing {removed} old network segments from input tile areas")
    return gdf_net[~mask].copy()


def snap_networks(gdf_new_net, gdf_orig_net, tolerance_m=3.0):
    """Snap endpoints of new network segments to nearby original endpoints.

    Works in a metric CRS (EPSG:3857) for distance calculations,
    then converts back to EPSG:4326.

    Parameters
    ----------
    gdf_new_net : GeoDataFrame — new network segments (EPSG:4326)
    gdf_orig_net : GeoDataFrame — original network segments (EPSG:4326)
    tolerance_m : float — snap distance in meters (default 3.0)
    """
    if len(gdf_new_net) == 0 or len(gdf_orig_net) == 0:
        return gdf_new_net

    # Project to metric CRS
    new_m = gdf_new_net.to_crs(epsg=3857)
    orig_m = gdf_orig_net.to_crs(epsg=3857)

    # Collect all endpoints from original network
    orig_points = []
    for geom in orig_m.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                orig_points.append(Point(part.coords[0]))
                orig_points.append(Point(part.coords[-1]))
        else:
            orig_points.append(Point(geom.coords[0]))
            orig_points.append(Point(geom.coords[-1]))

    if not orig_points:
        return gdf_new_net

    orig_points_union = unary_union(orig_points)

    snapped_geoms = []
    snap_count = 0
    for geom in new_m.geometry:
        if geom is None or geom.is_empty:
            snapped_geoms.append(geom)
            continue
        if geom.geom_type == "MultiLineString":
            # Snap each part separately
            from shapely.geometry import MultiLineString, LineString
            parts = []
            for part in geom.geoms:
                coords = list(part.coords)
                for idx in [0, -1]:
                    pt = Point(coords[idx])
                    nearest_pt = nearest_points(pt, orig_points_union)[1]
                    if pt.distance(nearest_pt) < tolerance_m:
                        coords[idx] = nearest_pt.coords[0]
                        snap_count += 1
                parts.append(LineString(coords))
            snapped_geoms.append(MultiLineString(parts))
        else:
            coords = list(geom.coords)
            for idx in [0, -1]:
                pt = Point(coords[idx])
                nearest_pt = nearest_points(pt, orig_points_union)[1]
                if pt.distance(nearest_pt) < tolerance_m:
                    coords[idx] = nearest_pt.coords[0]
                    snap_count += 1
            snapped_geoms.append(type(geom)(coords))

    print(f"  Snapped {snap_count} endpoints (tolerance={tolerance_m}m)")

    new_m = new_m.copy()
    new_m["geometry"] = snapped_geoms
    return new_m.to_crs(epsg=4326)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply trained model and produce global network + polygon GeoJSON."
    )

    # Tile ID source (one of these required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tile_ids", nargs="+",
                       help="Zoom-18 tile IDs (e.g., 79325_97025 79322_97010)")
    group.add_argument("--geojson",
                       help="GeoJSON file — tile IDs extracted from 'tile_id' column")

    # Model + data paths
    parser.add_argument("--model_path", default=os.getenv("MODEL_PATH", "./outputs/suggestion_model.pt"),
                        help="Path to trained model .pt")
    parser.add_argument("--tiles_dir", default=os.getenv("TILES_DIR"),
                        help="RGB satellite tiles directory")
    parser.add_argument("--t2n_dir", default=os.getenv("T2N_DIR"),
                        help="T2N rasterized polygon masks directory")
    parser.add_argument("--conf_dir", default=os.getenv("CONF_DIR"),
                        help="T2N confidence masks directory")

    # Original global files
    parser.add_argument("--original_polygons", default=os.getenv("ORIGINAL_POLYGONS"),
                        help="Original global polygon GeoJSON to update")
    parser.add_argument("--original_network", default=os.getenv("ORIGINAL_NETWORK"),
                        help="Original global network GeoJSON to update")

    # Outputs
    parser.add_argument("--output_polygons", default="./outputs/corrected_polygons_global.geojson",
                        help="Output global polygon GeoJSON path")
    parser.add_argument("--output_network", default="./outputs/corrected_network_global.geojson",
                        help="Output global network GeoJSON path")

    # Options
    parser.add_argument("--head", choices=["fix", "full"], default="fix",
                        help="Model head to use (default: fix)")
    parser.add_argument("--enable_remove", action="store_true",
                        help="Enable remove head (model was trained with add+remove)")
    parser.add_argument("--snap_tolerance", type=float, default=3.0,
                        help="Snap tolerance in meters for connecting networks (default: 3.0)")

    args = parser.parse_args()

    # Validate required paths
    for arg_name in ["tiles_dir", "t2n_dir", "conf_dir", "original_polygons", "original_network"]:
        if getattr(args, arg_name) is None:
            parser.error(f"--{arg_name} is required (set via CLI or .env)")

    # ── Step 1: Get zoom-18 tile IDs ──
    if args.geojson:
        print(f"Loading GeoJSON: {args.geojson}")
        gdf_input = gpd.read_file(args.geojson)
        tile_ids_18 = gdf_input["tile_id"].unique().tolist()
    else:
        tile_ids_18 = args.tile_ids

    print(f"  {len(tile_ids_18)} zoom-18 tiles")

    # ── Step 2: Expand to zoom-19 children ──
    tile_ids_19 = zoom18_to_zoom19(tile_ids_18)
    print(f"  {len(tile_ids_19)} zoom-19 children")

    # ── Step 3: Filter to tiles with existing data ──
    tile_ids_19 = filter_existing_tiles(
        tile_ids_19, args.tiles_dir, args.t2n_dir, args.conf_dir
    )
    print(f"  {len(tile_ids_19)} zoom-19 tiles with data")

    if not tile_ids_19:
        print("No valid tiles found. Check paths and tile IDs.")
        sys.exit(1)

    # ── Step 4: Load model ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    print(f"Loading model: {args.model_path} (enable_remove={args.enable_remove})")
    model = ResidualFixNet(enable_remove=args.enable_remove).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ── Step 5: Run inference ──
    print(f"\nRunning inference ({args.head} head) on {len(tile_ids_19)} tiles...")
    predictions = run_inference(
        model, tile_ids_19, device, args.head,
        tiles_dir=args.tiles_dir,
        t2n_dir=args.t2n_dir,
        conf_dir=args.conf_dir,
    )
    print(f"  {len(predictions)} masks produced")

    # ── Step 6: Re-polygonize ──
    print("\nRe-polygonizing predicted masks...")
    gdf_new_polys = polygonize_predictions(predictions)
    gdf_new_polys["n_suggestion"] = 0
    print(f"  {len(gdf_new_polys)} new polygons from {gdf_new_polys['tile_id'].nunique()} zoom-18 tiles")

    if len(gdf_new_polys) == 0:
        print("No polygons produced. Exiting.")
        sys.exit(1)

    # ── Step 7: Compute bbox and run topology on new polygons ──
    bbox = compute_bbox_from_tiles(tile_ids_18)
    print(f"\nBounding box: {bbox}")

    gdf_new_network = run_topology(gdf_new_polys, bbox)

    # ── Step 8: Load original global files ──
    print(f"\nLoading original polygons: {args.original_polygons}")
    gdf_orig_polys = gpd.read_file(args.original_polygons)
    print(f"  {len(gdf_orig_polys)} original polygons")

    print(f"Loading original network: {args.original_network}")
    gdf_orig_net = gpd.read_file(args.original_network)
    print(f"  {len(gdf_orig_net)} original network segments")

    # ── Step 9: Remove old data from input tiles ──
    print("\nRemoving old data from input tiles...")
    gdf_orig_polys = remove_tiles_from_polygons(gdf_orig_polys, tile_ids_18)
    gdf_orig_net = remove_tiles_from_network(gdf_orig_net, tile_ids_18)

    # ── Step 10: Snap new network endpoints to original network ──
    print("\nSnapping new network to original...")
    gdf_new_network = snap_networks(gdf_new_network, gdf_orig_net,
                                     tolerance_m=args.snap_tolerance)

    # ── Step 11: Merge and save ──
    print("\nMerging global files...")
    gdf_global_polys = pd.concat([gdf_orig_polys, gdf_new_polys], ignore_index=True)
    gdf_global_net = pd.concat([gdf_orig_net, gdf_new_network], ignore_index=True)

    out_poly = Path(args.output_polygons)
    out_poly.parent.mkdir(parents=True, exist_ok=True)
    gdf_global_polys.to_file(out_poly, driver="GeoJSON")
    print(f"  Global polygon GeoJSON saved to: {out_poly} ({len(gdf_global_polys)} polygons)")

    out_net = Path(args.output_network)
    out_net.parent.mkdir(parents=True, exist_ok=True)
    gdf_global_net.to_file(out_net, driver="GeoJSON")
    print(f"  Global network GeoJSON saved to: {out_net} ({len(gdf_global_net)} segments)")

    print(f"\nDone! {len(gdf_new_polys)} new polygons replaced in {len(tile_ids_18)} tiles. "
          f"{len(gdf_global_polys)} total polygons, {len(gdf_global_net)} total network edges.")


if __name__ == "__main__":
    main()
