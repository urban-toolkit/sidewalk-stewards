import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# .env is 4 levels up: stewards_scripts → boston → stewards_files → backend → project root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

import cv2
import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# Local imports — use HELPERS_PATH from env if set, otherwise fall back to sibling dir
_helpers = os.getenv("HELPERS_PATH") or str(Path(__file__).parent / "helper_scripts")
sys.path.insert(0, _helpers)
from polygon_fixing import get_tile_bounds, rasterize_polygons_to_mask
from tile2net_training_utils import ResidualFixNet, train_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def deg2num(lon, lat, zoom):
    """Convert lon/lat to slippy-map tile x/y at a given zoom level."""
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return xtile, ytile

def find_covered_tile_ids(gdf, tiles_dir, t2n_dir, conf_dir):
    """Derive zoom-20 tile IDs from the tile_id property (zoom-19 parents) in the GeoJSON.

    Each parent tile_id x_y has four zoom-20 children:
      2x_2y, (2x+1)_2y, 2x_(2y+1), (2x+1)_(2y+1)
    Only children for which all three required files exist are returned.
    """
    tiles_dir = Path(tiles_dir)
    t2n_dir   = Path(t2n_dir)
    conf_dir  = Path(conf_dir)

    parent_ids = gdf["tile_id"].dropna().unique().tolist()
    print(f"  parent tile_ids from GeoJSON: {parent_ids}")

    tile_ids = []
    for pid in parent_ids:
        px, py = map(int, pid.split("_"))
        children = [
            f"{2*px}_{2*py}",
            f"{2*px+1}_{2*py}",
            f"{2*px}_{2*py+1}",
            f"{2*px+1}_{2*py+1}",
        ]
        print(f"  {pid} -> children: {children}")
        for tid in children:
            if not (tiles_dir / f"{tid}.jpg").exists():
                print(f"    missing RGB:  {tid}.jpg")
                continue
            if not (t2n_dir / f"{tid}.png").exists():
                print(f"    missing T2N:  {tid}.png")
                continue
            if not (conf_dir / f"{tid}.png").exists():
                print(f"    missing conf: {tid}.png")
                continue
            tile_ids.append(tid)

    return tile_ids


def rasterize_suggestions_to_gt(gdf, tile_ids, gt_dir):
    """Rasterize input polygons to per-tile 256x256 PNG masks."""
    gt_dir = Path(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for tid in tqdm(tile_ids, desc="Rasterizing suggestions"):
        x, y = map(int, tid.split("_"))
        mask = rasterize_polygons_to_mask(gdf, x, y, zoom=20, resolution=256)
        Image.fromarray(mask, mode="L").save(gt_dir / f"{tid}.png")

def run_inference(model, tile_ids, device, head, tiles_dir, t2n_dir, conf_dir):
    """Run trained model on tiles and return predicted masks.

    Based on the inference loop in tile2net_polygon_fixing.ipynb.

    Parameters
    ----------
    model : ResidualFixNet
        Trained model in eval mode.
    tile_ids : list of str
    device : torch.device
    head : str
        "fix" or "full".
    tiles_dir, t2n_dir, conf_dir : str or Path
        Directories for RGB, T2N masks, confidence masks.

    Returns
    -------
    dict : tile_id → uint8 mask (256x256), 0 or 255.
    """
    tiles_dir = Path(tiles_dir)
    t2n_dir = Path(t2n_dir)
    conf_dir = Path(conf_dir)

    model.eval()
    predictions = {}

    for tid in tqdm(tile_ids, desc="Running inference"):
        # Load RGB
        sat = np.array(Image.open(tiles_dir / f"{tid}.jpg").convert("RGB"))
        rgb = sat.astype(np.float32) / 255.0

        # Load confidence
        conf_path = conf_dir / f"{tid}.png"
        if not conf_path.exists():
            continue
        conf_ch = np.array(Image.open(conf_path).convert("L")).astype(np.float32) / 255.0

        # Load T2N polygon mask
        poly_mask_path = t2n_dir / f"{tid}.png"
        if not poly_mask_path.exists():
            continue
        poly_mask = np.array(Image.open(poly_mask_path).convert("L"))
        t2n_ch = (poly_mask > 127).astype(np.float32)

        # Build 5-channel input: (1, 5, 256, 256)
        input_np = np.concatenate([
            rgb,
            t2n_ch[:, :, None],
            conf_ch[:, :, None],
        ], axis=2).transpose(2, 0, 1)

        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            if len(outputs) == 3:
                pred_full, pred_fix, pred_remove = outputs
            else:
                pred_full, pred_fix = outputs
                pred_remove = None

        if head == "fix":
            pred_np = pred_fix[0, 0].cpu().numpy()
            t2n_binary = t2n_ch > 0.5
            if pred_remove is not None:
                pred_remove_np = pred_remove[0, 0].cpu().numpy()
                corrected = ((t2n_binary & ~(pred_remove_np > 0.5)) | (pred_np > 0.5)).astype(np.uint8) * 255
            else:
                corrected = (t2n_binary | (pred_np > 0.5)).astype(np.uint8) * 255
        else:
            pred_np = pred_full[0, 0].cpu().numpy()
            corrected = (pred_np > 0.5).astype(np.uint8) * 255

        predictions[tid] = corrected

    return predictions

# ─────────────────────────────────────────────────────────────────────────────
# Re-polygonization (from tile2net_polygon_fixing.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_polygons(geom):
    """Extract Polygon(s) from any Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polys = []
        for part in geom.geoms:
            polys.extend(_extract_polygons(part))
        return polys
    return []

def mask_to_polygons(mask, xtile, ytile, zoom=19, resolution=256,
                     simplify_tolerance=0.5, min_area_px=25,
                     morph_kernel=3):
    """Convert a binary mask (256x256) back to geo-referenced polygons.

    Parameters
    ----------
    mask : ndarray (256, 256) uint8, 0/255
    xtile, ytile : int — tile coordinates
    zoom : int
    resolution : int
    simplify_tolerance : float — Douglas-Peucker tolerance in pixels
    min_area_px : int — drop polygons smaller than this
    morph_kernel : int — morphological closing kernel size (0 to skip)

    Returns
    -------
    list of shapely Polygon in EPSG:4326
    """
    binary = (mask > 127).astype(np.uint8)

    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    west, south, east, north = get_tile_bounds(xtile, ytile, zoom)
    px_to_lon = (east - west) / resolution
    px_to_lat = (south - north) / resolution  # negative: y down in pixels

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue

        coords = contour.squeeze(1).astype(float)
        geo_coords = [
            (west + x * px_to_lon, north + y * px_to_lat) for x, y in coords
        ]

        try:
            poly = Polygon(geo_coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            valid_polys = _extract_polygons(poly)
            for vp in valid_polys:
                if simplify_tolerance > 0:
                    geo_tol = simplify_tolerance * abs(px_to_lon)
                    vp = vp.simplify(geo_tol)
                if not vp.is_empty and vp.area > 0:
                    polygons.append(vp)
        except Exception:
            continue

    return polygons

def polygonize_predictions(predictions):
    """Convert predicted masks to a GeoDataFrame with zoom-18 tile_id.

    Parameters
    ----------
    predictions : dict
        tile_id (zoom-19) → uint8 mask (256x256).

    Returns
    -------
    GeoDataFrame with columns: geometry, tile_id (zoom-18).
    """
    all_polys = []
    all_tile_ids = []

    for tid, mask in tqdm(predictions.items(), desc="Re-polygonizing"):
        if mask.max() == 0:
            continue
        xtile, ytile = map(int, tid.split("_"))
        polys = mask_to_polygons(mask, xtile, ytile)
        # Zoom-18 parent tile
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a sidewalk fix model from polygon suggestions (GeoJSON via stdin)."
    )
    parser.add_argument("--tiles_dir",     default=os.getenv("TILES_DIR"),
                        help="RGB satellite tiles directory")
    parser.add_argument("--t2n_dir",       default=os.getenv("T2N_DIR"),
                        help="T2N rasterized polygon masks directory")
    parser.add_argument("--conf_dir",      default=os.getenv("CONF_DIR"),
                        help="T2N confidence masks directory")
    parser.add_argument("--model_output",  default=os.getenv("TRAINED_MODEL_OUTPUT", "./outputs/suggestion_model.pt"),
                        help="Output model .pt path")
    parser.add_argument("--epochs",        type=int, default=200,
                        help="Training epochs (default: 200)")
    parser.add_argument("--head",          choices=["fix", "full"], default="fix",
                        help="Model head to use (default: fix)")
    parser.add_argument("--enable_remove", action="store_true",
                        help="Enable remove head (add+remove mode)")
    args = parser.parse_args()

    for arg_name in ["tiles_dir", "t2n_dir", "conf_dir"]:
        if getattr(args, arg_name) is None:
            parser.error(f"--{arg_name} is required (set via CLI or .env)")

    # ── Step 1: Read GeoJSON from stdin ──
    print("Reading GeoJSON from stdin...")
    raw = sys.stdin.read()
    if not raw.strip():
        print("Error: no GeoJSON received on stdin.", file=sys.stderr)
        sys.exit(1)
    gdf = gpd.read_file(io.StringIO(raw), driver="GeoJSON")
    print(f"  {len(gdf)} polygons loaded")

    # ── Step 2: Find covered tile IDs ──
    print("\nFinding covered zoom-20 tiles...")
    tile_ids = find_covered_tile_ids(
        gdf,
        tiles_dir=args.tiles_dir,
        t2n_dir=args.t2n_dir,
        conf_dir=args.conf_dir,
    )
    print(f"  {len(tile_ids)} tiles with data found")

    if not tile_ids:
        print("No tiles found. Check input paths and GeoJSON coverage.")
        sys.exit(1)

    # ── Step 3: Rasterize suggestions → GT masks ──
    with tempfile.TemporaryDirectory(prefix="suggestion_gt_") as gt_dir:
        print(f"\nRasterizing suggestions to: {gt_dir}")
        rasterize_suggestions_to_gt(gdf, tile_ids, gt_dir)

        # ── Step 4: Filter to tiles where GT differs from T2N ──
        t2n_dir_path = Path(args.t2n_dir)
        gt_dir_path  = Path(gt_dir)
        changed_tiles = []
        unchanged = 0
        for tid in tile_ids:
            t2n_path = t2n_dir_path / f"{tid}.png"
            gt_path  = gt_dir_path  / f"{tid}.png"
            if not t2n_path.exists() or not gt_path.exists():
                continue
            t2n_mask = np.array(Image.open(t2n_path).convert("L"))
            gt_mask  = np.array(Image.open(gt_path).convert("L"))
            if np.array_equal(t2n_mask, gt_mask):
                unchanged += 1
            else:
                changed_tiles.append(tid)

        print(f"  {len(changed_tiles)} tiles with changes, {unchanged} unchanged (skipped)")
        train_tile_ids = changed_tiles if changed_tiles else tile_ids
        if not changed_tiles:
            print("  Warning: no tiles differ -- training on all tiles")

        # ── Step 5: Select device ──
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"\nUsing device: {device}")

        # ── Step 6: Train model ──
        print(f"\nTraining ResidualFixNet ({args.epochs} epochs, head={args.head})...")
        model = ResidualFixNet(enable_remove=args.enable_remove).to(device)
        n = len(train_tile_ids)
        bucket_info = {"all": {"total": n, "train": n, "val": n}}

        history, _, _, _ = train_model(
            model,
            train_tile_ids=train_tile_ids,
            val_ids=train_tile_ids,
            bucket_info=bucket_info,
            device=device,
            n_epochs=args.epochs,
            gt_dir=gt_dir,
            t2n_dir=args.t2n_dir,
            conf_dir=args.conf_dir,
            tiles_dir=args.tiles_dir,
        )

        # ── Step 7: Save model ──
        model_path = Path(args.model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()