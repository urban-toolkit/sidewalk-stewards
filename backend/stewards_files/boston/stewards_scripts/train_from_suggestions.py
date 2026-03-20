import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# .env is 4 levels up: stewards_scripts → boston → stewards_files → backend → project root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Local imports — use HELPERS_PATH from env if set, otherwise fall back to sibling dir
_helpers = os.getenv("HELPERS_PATH") or str(Path(__file__).parent / "helper_scripts")
sys.path.insert(0, _helpers)
from polygon_fixing import rasterize_polygons_to_mask
from tile2net_training_utils import ResidualFixNet, train_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--model_output",  default=os.getenv("TRANED_MODEL_OUTPUT", "./outputs/suggestion_model.pt"),
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