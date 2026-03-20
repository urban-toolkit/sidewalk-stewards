"""
Tile2Net vs Ground Truth — utility functions for loading, comparing,
and visualising sidewalk prediction masks.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

# ──────────────────────────────────────────────
# Paths & tile-ID discovery
# ──────────────────────────────────────────────

# Default paths — standardized dorchester_exports folder.
# All masks: {tile_id}.png  (255 = sidewalk, 0 = background)
# Satellite: {tile_id}.jpg
# Confidence: {tile_id}.png  (single-channel grayscale, 0–255)



def get_all_tile_ids(tiles_dir=None, t2n_dir=None, conf_dir=None, gt_dir=None):
    """Return a sorted list of tile IDs present in all four folders.

    Assumes standardized naming: {tile_id}.jpg for satellite,
    {tile_id}.png for all masks.
    """
    tiles_dir = tiles_dir
    t2n_dir = t2n_dir
    conf_dir = conf_dir 
    gt_dir = gt_dir 

    tile_ids_from_tiles = {
        f.replace(".jpg", "") for f in os.listdir(tiles_dir) if f.endswith(".jpg")
    }
    tile_ids_from_t2n = {
        f.replace(".png", "") for f in os.listdir(t2n_dir) if f.endswith(".png")
    }
    tile_ids_from_conf = {
        f.replace(".png", "") for f in os.listdir(conf_dir) if f.endswith(".png")
    }
    tile_ids_from_gt = {
        f.replace(".png", "") for f in os.listdir(gt_dir) if f.endswith(".png")
    }
    return sorted(
        tile_ids_from_tiles & tile_ids_from_t2n & tile_ids_from_conf & tile_ids_from_gt
    )


ALL_TILE_IDS = get_all_tile_ids()


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_tile_data(tile_id, tiles_dir=None, t2n_dir=None,
                   conf_dir=None, gt_dir=None):
    """Load all four images for a given tile_id and return as numpy arrays.

    Assumes standardized format:
        - Satellite:  {tile_id}.jpg  (RGB)
        - T2N mask:   {tile_id}.png  (grayscale, 255 = sidewalk)
        - Confidence: {tile_id}.png  (grayscale, 0–255)
        - GT mask:    {tile_id}.png  (grayscale, 255 = sidewalk)

    Parameters
    ----------
    tile_id   : str
    tiles_dir : str or None — satellite folder (default: dorchester_exports/tiles)
    t2n_dir   : str or None — T2N mask folder (default: dorchester_exports/masks_tile2net)
    conf_dir  : str or None — confidence folder (default: dorchester_exports/masks_confidence)
    gt_dir    : str or None — ground truth folder (default: dorchester_exports/masks_groundtruth_polygons)

    Returns
    -------
    satellite  : (256, 256, 3) uint8 RGB
    confidence : (256, 256) float32 in [0, 1]
    t2n_mask   : (256, 256) bool — True = sidewalk
    gt_mask    : (256, 256) bool — True = sidewalk
    """
    tiles_dir = tiles_dir 
    t2n_dir = t2n_dir 
    conf_dir = conf_dir 
    gt_dir = gt_dir 

    satellite = np.array(
        Image.open(os.path.join(tiles_dir, f"{tile_id}.jpg")).convert("RGB")
    )

    conf_raw = np.array(
        Image.open(os.path.join(conf_dir, f"{tile_id}.png")).convert("L")
    )
    confidence = conf_raw.astype(np.float32) / 255.0

    t2n_raw = np.array(
        Image.open(os.path.join(t2n_dir, f"{tile_id}.png")).convert("L")
    )
    t2n_mask = t2n_raw > 127  # bool: True = sidewalk

    gt_raw = np.array(
        Image.open(os.path.join(gt_dir, f"{tile_id}.png")).convert("L")
    )
    gt_mask = gt_raw > 127  # bool: True = sidewalk

    return satellite, confidence, t2n_mask, gt_mask


# ──────────────────────────────────────────────
# Metrics & overlays
# ──────────────────────────────────────────────

def compute_difference_overlay(t2n_mask, gt_mask, show_overlap=True):
    """Build an RGB difference image.

    Colors:
        Green  (0,200,0)   — overlap: both predict sidewalk (TP)
        Red    (200,0,0)   — missed ground truth: GT=1, T2N=0 (FN)
        Yellow (200,200,0) — over-predicted by T2N: GT=0, T2N=1 (FP)
        Dark   (30,30,30)  — both background (TN)
    """
    h, w = t2n_mask.shape
    overlay = np.full((h, w, 3), 30, dtype=np.uint8)  # dark background

    if show_overlap:
        # True Positive — green
        tp = t2n_mask & gt_mask
        overlay[tp] = [0, 200, 0]

    # False Negative — red (missed by T2N)
    fn = ~t2n_mask & gt_mask
    overlay[fn] = [200, 0, 0]

    # False Positive — yellow (over-predicted by T2N)
    fp = t2n_mask & ~gt_mask
    overlay[fp] = [200, 200, 0]

    return overlay


def compute_iou(t2n_mask, gt_mask):
    """Compute Intersection over Union for the sidewalk class."""
    intersection = np.sum(t2n_mask & gt_mask)
    union = np.sum(t2n_mask | gt_mask)
    if union == 0:
        return float("nan")
    return intersection / union


def compute_recall(t2n_mask, gt_mask):
    """Compute recall (sensitivity) for the sidewalk class.

    Recall = TP / (TP + FN)
    "Of all GT sidewalk pixels, what fraction did T2N find?"
    """
    tp = np.sum(t2n_mask & gt_mask)
    total_gt = np.sum(gt_mask)
    if total_gt == 0:
        return float("nan")
    return tp / total_gt


# ──────────────────────────────────────────────
# Generic error-pixel filters (work for FN or FP)
# ──────────────────────────────────────────────

def _filter_erosion(error_mask, kernel_size=5):
    """Remove thin strips via morphological erosion.

    Error pixels that disappear after eroding with a (kernel_size x kernel_size)
    structuring element are considered misalignment artifacts.
    Returns the eroded mask (only 'thick' regions survive).
    """
    struct = np.ones((kernel_size, kernel_size), dtype=bool)
    return ndimage.binary_erosion(error_mask, structure=struct)


def _filter_small_components(error_mask, min_area=50):
    """Remove small connected blobs below *min_area* pixels."""
    labeled, n_components = ndimage.label(error_mask)
    if n_components == 0:
        return error_mask.copy()
    component_sizes = ndimage.sum(error_mask, labeled, range(1, n_components + 1))
    filtered = error_mask.copy()
    for idx, size in enumerate(component_sizes, start=1):
        if size < min_area:
            filtered[labeled == idx] = False
    return filtered


def _filter_low_reference(error_mask, reference_mask, min_pixels=100):
    """If the reference region is very small, discard all error pixels as noise.

    For FN: reference_mask = gt_mask  (few GT pixels → FN is noise)
    For FP: reference_mask = t2n_mask (few T2N pixels → FP is noise)
    """
    if reference_mask.sum() < min_pixels:
        return np.zeros_like(error_mask)
    return error_mask.copy()


def _filter_border(error_mask, border_px=5):
    """Zero out error pixels within *border_px* of the tile edge."""
    filtered = error_mask.copy()
    filtered[:border_px, :] = False
    filtered[-border_px:, :] = False
    filtered[:, :border_px] = False
    filtered[:, -border_px:] = False
    return filtered


def _filter_distance_to_tp(error_mask, tp_mask, max_dist=5):
    """Remove error pixels that are within *max_dist* pixels of a TP pixel.

    Uses the Euclidean distance transform on the inverted TP mask so that
    each pixel gets its distance to the nearest TP region.
    """
    if tp_mask.sum() == 0:
        return error_mask.copy()
    dist_from_tp = ndimage.distance_transform_edt(~tp_mask)
    filtered = error_mask.copy()
    filtered[dist_from_tp <= max_dist] = False
    return filtered


def _filter_combined(error_mask, reference_mask, tp_mask,
                     erosion_kernel=5, min_component_area=50,
                     min_ref_pixels=100, border_px=5, tp_max_dist=5):
    """Apply all 5 filters in parallel and take the union of removals.

    Each filter runs independently on the original error_mask.  A pixel is
    removed if ANY filter says it is irrelevant.

    Parameters
    ----------
    error_mask : bool — the error pixels to filter (FN or FP)
    reference_mask : bool — the "source" mask for the low-pixel check
        (gt_mask for FN, t2n_mask for FP)
    tp_mask : bool — true positive mask for distance filter

    Returns:
        surviving : bool mask — error pixels that survived all filters
        removed   : bool mask — error pixels removed by at least one filter
    """
    survived_erosion = _filter_erosion(error_mask, kernel_size=erosion_kernel)
    survived_components = _filter_small_components(error_mask, min_area=min_component_area)
    survived_low_ref = _filter_low_reference(error_mask, reference_mask, min_pixels=min_ref_pixels)
    survived_border = _filter_border(error_mask, border_px=border_px)
    survived_dist = _filter_distance_to_tp(error_mask, tp_mask, max_dist=tp_max_dist)

    removed_erosion = error_mask & ~survived_erosion
    removed_components = error_mask & ~survived_components
    removed_low_ref = error_mask & ~survived_low_ref
    removed_border = error_mask & ~survived_border
    removed_dist = error_mask & ~survived_dist

    removed = (removed_erosion | removed_components | removed_low_ref
               | removed_border | removed_dist)
    surviving = error_mask & ~removed

    return surviving, removed


# ──────────────────────────────────────────────
# Public FN filter API (backward-compatible)
# ──────────────────────────────────────────────

def filter_fn_erosion(fn_mask, kernel_size=5):
    """Remove thin FN strips via morphological erosion."""
    return _filter_erosion(fn_mask, kernel_size=kernel_size)

def filter_fn_small_components(fn_mask, min_area=50):
    """Remove small connected FN blobs below *min_area* pixels."""
    return _filter_small_components(fn_mask, min_area=min_area)

def filter_fn_low_gt(fn_mask, gt_mask, min_gt_pixels=100):
    """If total GT sidewalk area is very small, discard all FN as noise."""
    return _filter_low_reference(fn_mask, gt_mask, min_pixels=min_gt_pixels)

def filter_fn_border(fn_mask, border_px=5):
    """Zero out FN pixels within *border_px* of the tile edge."""
    return _filter_border(fn_mask, border_px=border_px)

def filter_fn_distance_to_tp(fn_mask, tp_mask, max_dist=5):
    """Remove FN pixels that are within *max_dist* pixels of a TP pixel."""
    return _filter_distance_to_tp(fn_mask, tp_mask, max_dist=max_dist)

def filter_fn_combined(fn_mask, gt_mask, tp_mask,
                       erosion_kernel=5, min_component_area=50,
                       min_gt_pixels=100, border_px=5, tp_max_dist=5):
    """Apply all 5 FN filters in parallel (union of removals).

    Returns:
        surviving_fn : bool mask — FN pixels that survived all filters
        fn_removed   : bool mask — FN pixels removed by at least one filter
    """
    return _filter_combined(fn_mask, gt_mask, tp_mask,
                            erosion_kernel=erosion_kernel,
                            min_component_area=min_component_area,
                            min_ref_pixels=min_gt_pixels,
                            border_px=border_px,
                            tp_max_dist=tp_max_dist)


# ──────────────────────────────────────────────
# Public FP filter API (symmetric to FN)
# ──────────────────────────────────────────────

def filter_fp_combined(fp_mask, t2n_mask, tp_mask,
                       erosion_kernel=5, min_component_area=50,
                       min_t2n_pixels=100, border_px=5, tp_max_dist=5):
    """Apply all 5 FP filters in parallel (union of removals).

    Symmetric to filter_fn_combined but uses t2n_mask as the reference
    for the low-pixel check (few T2N pixels → FP is noise).

    Returns:
        surviving_fp : bool mask — FP pixels that survived all filters
        fp_removed   : bool mask — FP pixels removed by at least one filter
    """
    return _filter_combined(fp_mask, t2n_mask, tp_mask,
                            erosion_kernel=erosion_kernel,
                            min_component_area=min_component_area,
                            min_ref_pixels=min_t2n_pixels,
                            border_px=border_px,
                            tp_max_dist=tp_max_dist)


# ──────────────────────────────────────────────
# Adjusted metrics (after filtering)
# ──────────────────────────────────────────────

def compute_adjusted_iou(t2n_mask, gt_mask, fn_removed, fp_removed=None):
    """IoU after 'forgiving' removed FN (and optionally FP) pixels.

    adjusted_gt  = gt_mask  AND NOT fn_removed   (forgive missed GT)
    adjusted_t2n = t2n_mask AND NOT fp_removed    (forgive over-prediction)
    IoU = |adj_T2N ∩ adj_GT| / |adj_T2N ∪ adj_GT|.
    """
    adjusted_gt = gt_mask & ~fn_removed
    adjusted_t2n = t2n_mask & ~fp_removed if fp_removed is not None else t2n_mask
    intersection = np.sum(adjusted_t2n & adjusted_gt)
    union = np.sum(adjusted_t2n | adjusted_gt)
    if union == 0:
        return float("nan")
    return intersection / union


def compute_adjusted_recall(t2n_mask, gt_mask, fn_removed, fp_removed=None):
    """Recall after 'forgiving' removed FN (and optionally FP) pixels.

    adjusted_gt  = gt_mask  AND NOT fn_removed
    adjusted_t2n = t2n_mask AND NOT fp_removed
    Recall = |adj_T2N ∩ adj_GT| / |adj_GT|.
    """
    adjusted_gt = gt_mask & ~fn_removed
    adjusted_t2n = t2n_mask & ~fp_removed if fp_removed is not None else t2n_mask
    tp = np.sum(adjusted_t2n & adjusted_gt)
    total_adj_gt = np.sum(adjusted_gt)
    if total_adj_gt == 0:
        return float("nan")
    return tp / total_adj_gt


# ──────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────

def build_filtered_overlay(t2n_mask, gt_mask, filtered_fn, filtered_fp=None):
    """Like compute_difference_overlay but uses pre-filtered FN/FP masks.

    Colors:
        Green  — TP (overlap)
        Red    — remaining (relevant) FN
        Yellow — remaining (relevant) FP
        Dark   — background + removed FN + removed FP

    Parameters
    ----------
    filtered_fn : bool mask — surviving FN pixels (after filtering)
    filtered_fp : bool mask or None — surviving FP pixels (after filtering).
        If None, all FP pixels are shown (no FP filtering).
    """
    h, w = t2n_mask.shape
    overlay = np.full((h, w, 3), 30, dtype=np.uint8)

    tp = t2n_mask & gt_mask
    overlay[tp] = [0, 200, 0]

    overlay[filtered_fn] = [200, 0, 0]

    if filtered_fp is not None:
        overlay[filtered_fp] = [200, 200, 0]
    else:
        fp = t2n_mask & ~gt_mask
        overlay[fp] = [200, 200, 0]

    return overlay


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def _fmt_metric(value):
    """Format a float metric, handling NaN."""
    return f"{value:.3f}" if not np.isnan(value) else "N/A"


def _fmt_gain(new_val, orig_val):
    """Format the difference between two metrics as a signed string."""
    if np.isnan(new_val) or np.isnan(orig_val):
        return "N/A"
    return f"{new_val - orig_val:+.3f}"


def display_comparison(tile_ids=None, n_random=3, figscale=3.0,
                       erosion_kernel=5, min_component_area=50,
                       min_gt_pixels=100, border_px=5, tp_max_dist=5,
                       tiles_dir=None, t2n_dir=None,
                       conf_dir=None, gt_dir=None):
    """Display a 6-column comparison for one or more tiles.

    Columns: Satellite | Confidence (hot) | Tile2Net Mask | Ground Truth |
             Difference (original) | Filtered Difference

    Parameters
    ----------
    tile_ids : str, list of str, or None
        Specific tile ID(s) to display. If None, pick randomly.
    n_random : int
        Number of random tiles to display when tile_ids is None.
    figscale : float
        Controls the size of each subplot cell.
    erosion_kernel, min_component_area, min_gt_pixels, border_px, tp_max_dist
        Hyper-parameters for the combined FN filter.
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths (passed to load_tile_data).
    """
    # Resolve tile list
    if tile_ids is None:
        ids = np.random.choice(ALL_TILE_IDS, min(n_random, len(ALL_TILE_IDS)))
    elif isinstance(tile_ids, str):
        ids = [tile_ids]
    else:
        ids = list(tile_ids)

    n_rows = len(ids)
    col_titles = [
        "Satellite", "Confidence (hot)", "Tile2Net Mask",
        "Ground Truth", "Difference", "Filtered Difference",
    ]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figscale * n_cols, figscale * n_rows),
        squeeze=False,
    )

    # Difference legend patches
    legend_patches = [
        mpatches.Patch(color=np.array([0, 200, 0]) / 255, label="Overlap (TP)"),
        mpatches.Patch(color=np.array([200, 0, 0]) / 255, label="Missed GT (FN)"),
        mpatches.Patch(color=np.array([200, 200, 0]) / 255, label="Over-pred (FP)"),
        mpatches.Patch(color=np.array([30, 30, 30]) / 255, label="Background (TN)"),
    ]

    for row_idx, tid in enumerate(ids):
        satellite, confidence, t2n_mask, gt_mask = load_tile_data(
            tid, tiles_dir=tiles_dir, t2n_dir=t2n_dir,
            conf_dir=conf_dir, gt_dir=gt_dir)
        diff_overlay = compute_difference_overlay(t2n_mask, gt_mask)

        # Original metrics
        iou = compute_iou(t2n_mask, gt_mask)
        recall = compute_recall(t2n_mask, gt_mask)

        # Combined FN filter
        fn_mask = ~t2n_mask & gt_mask
        fp_mask = t2n_mask & ~gt_mask
        tp_mask = t2n_mask & gt_mask
        surviving_fn, fn_removed = filter_fn_combined(
            fn_mask, gt_mask, tp_mask,
            erosion_kernel=erosion_kernel,
            min_component_area=min_component_area,
            min_gt_pixels=min_gt_pixels,
            border_px=border_px,
            tp_max_dist=tp_max_dist,
        )
        # Combined FP filter (symmetric)
        surviving_fp, fp_removed = filter_fp_combined(
            fp_mask, t2n_mask, tp_mask,
            erosion_kernel=erosion_kernel,
            min_component_area=min_component_area,
            min_t2n_pixels=min_gt_pixels,
            border_px=border_px,
            tp_max_dist=tp_max_dist,
        )
        filtered_overlay = build_filtered_overlay(
            t2n_mask, gt_mask, surviving_fn, surviving_fp,
        )

        # Adjusted metrics (forgiving both FN and FP)
        adj_iou = compute_adjusted_iou(t2n_mask, gt_mask, fn_removed, fp_removed)
        adj_recall = compute_adjusted_recall(t2n_mask, gt_mask, fn_removed, fp_removed)

        ax_sat, ax_conf, ax_t2n, ax_gt, ax_diff, ax_filt = axes[row_idx]

        # Satellite
        ax_sat.imshow(satellite)
        ax_sat.set_ylabel(tid, fontsize=9, rotation=90, labelpad=10, va="center")

        # Confidence — hot colormap
        ax_conf.imshow(confidence, cmap="hot", vmin=0, vmax=1)

        # Tile2Net mask (white = sidewalk)
        ax_t2n.imshow(t2n_mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)

        # Ground truth mask (white = sidewalk)
        ax_gt.imshow(gt_mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)

        # Difference overlay — original IoU + recall
        ax_diff.imshow(diff_overlay)
        ax_diff.set_xlabel(
            f"IoU: {_fmt_metric(iou)}  Recall: {_fmt_metric(recall)}",
            fontsize=8,
        )

        # Filtered difference — adjusted IoU + recall with gains
        ax_filt.imshow(filtered_overlay)
        ax_filt.set_xlabel(
            f"IoU: {_fmt_metric(adj_iou)} ({_fmt_gain(adj_iou, iou)})  "
            f"Recall: {_fmt_metric(adj_recall)} ({_fmt_gain(adj_recall, recall)})",
            fontsize=8,
        )

        # Column titles on first row only
        if row_idx == 0:
            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold")

    # Turn off all ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06 if n_rows <= 3 else 0.03)
    plt.show()


def display_fn_filters(
    tile_id=None,
    # filter hyper-parameters
    erosion_kernel=5,
    min_component_area=50,
    min_gt_pixels=100,
    border_px=5,
    tp_max_dist=5,
    figscale=3.2,
    tiles_dir=None, t2n_dir=None,
    conf_dir=None, gt_dir=None,
):
    """Show how each of the 5 FN-filtering methods affects a single tile.

    Parameters
    ----------
    tile_id : str or None
        Tile to inspect. If None a random tile is chosen.
    erosion_kernel, min_component_area, min_gt_pixels, border_px, tp_max_dist
        Hyper-parameters for the five filters.
    figscale : float
        Size scaling for subplots.
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths (passed to load_tile_data).
    """
    if tile_id is None:
        tile_id = np.random.choice(ALL_TILE_IDS)

    satellite, confidence, t2n_mask, gt_mask = load_tile_data(
        tile_id, tiles_dir=tiles_dir, t2n_dir=t2n_dir,
        conf_dir=conf_dir, gt_dir=gt_dir)

    # Raw masks
    fn_mask = ~t2n_mask & gt_mask     # original false negatives
    tp_mask = t2n_mask & gt_mask      # true positives
    original_iou = compute_iou(t2n_mask, gt_mask)
    original_recall = compute_recall(t2n_mask, gt_mask)

    # Apply each filter — returns the *surviving* FN pixels
    filters = [
        ("Erosion\n(kernel={})".format(erosion_kernel),
         filter_fn_erosion(fn_mask, kernel_size=erosion_kernel)),
        ("Small blobs\n(area<{})".format(min_component_area),
         filter_fn_small_components(fn_mask, min_area=min_component_area)),
        ("Low GT\n(<{} px)".format(min_gt_pixels),
         filter_fn_low_gt(fn_mask, gt_mask, min_gt_pixels=min_gt_pixels)),
        ("Border mask\n({}px)".format(border_px),
         filter_fn_border(fn_mask, border_px=border_px)),
        ("Dist to TP\n(≤{}px)".format(tp_max_dist),
         filter_fn_distance_to_tp(fn_mask, tp_mask, max_dist=tp_max_dist)),
    ]

    n_cols = len(filters) + 1  # +1 for original difference
    fig, axes = plt.subplots(1, n_cols, figsize=(figscale * n_cols, figscale * 1.55),
                             squeeze=False)
    axes = axes[0]

    # --- Column 0: original difference overlay ---
    orig_overlay = compute_difference_overlay(t2n_mask, gt_mask, show_overlap=True)
    axes[0].imshow(orig_overlay)
    axes[0].set_title("Original Diff", fontsize=10, fontweight="bold")
    axes[0].set_xlabel(
        f"IoU: {_fmt_metric(original_iou)}  Recall: {_fmt_metric(original_recall)}",
        fontsize=8,
    )
    axes[0].set_ylabel(tile_id, fontsize=9, rotation=90, labelpad=10, va="center")

    # --- Columns 1-5: each filter ---
    for col, (label, surviving_fn) in enumerate(filters, start=1):
        fn_removed = fn_mask & ~surviving_fn  # pixels the filter discarded

        overlay = build_filtered_overlay(t2n_mask, gt_mask, surviving_fn)
        adj_iou = compute_adjusted_iou(t2n_mask, gt_mask, fn_removed)
        adj_recall = compute_adjusted_recall(t2n_mask, gt_mask, fn_removed)

        axes[col].imshow(overlay)
        axes[col].set_title(label, fontsize=9, fontweight="bold")

        n_removed = int(fn_removed.sum())
        n_orig_fn = int(fn_mask.sum())
        pct_s = f"{n_removed / n_orig_fn * 100:.0f}%" if n_orig_fn > 0 else "–"

        axes[col].set_xlabel(
            f"IoU: {_fmt_metric(adj_iou)} ({_fmt_gain(adj_iou, original_iou)})  "
            f"Recall: {_fmt_metric(adj_recall)} ({_fmt_gain(adj_recall, original_recall)})\n"
            f"Removed {n_removed}/{n_orig_fn} FN ({pct_s})",
            fontsize=7,
        )

    # Legend & cleanup
    legend_patches = [
        mpatches.Patch(color=np.array([0, 200, 0]) / 255, label="Overlap (TP)"),
        mpatches.Patch(color=np.array([200, 0, 0]) / 255, label="Remaining FN"),
        mpatches.Patch(color=np.array([200, 200, 0]) / 255, label="Over-pred (FP)"),
        mpatches.Patch(color=np.array([30, 30, 30]) / 255, label="BG / Removed FN"),
    ]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.show()

    # Print a compact summary table
    print(f"\n{'Method':<28} {'Adj IoU':>8} {'IoU Δ':>8} {'Adj Rec':>8} {'Rec Δ':>8} {'FN removed':>12} {'%':>6}")
    print("─" * 82)
    for label, surviving_fn in filters:
        fn_removed = fn_mask & ~surviving_fn
        adj_iou = compute_adjusted_iou(t2n_mask, gt_mask, fn_removed)
        adj_recall = compute_adjusted_recall(t2n_mask, gt_mask, fn_removed)
        iou_gain = adj_iou - original_iou if not (np.isnan(adj_iou) or np.isnan(original_iou)) else float("nan")
        rec_gain = adj_recall - original_recall if not (np.isnan(adj_recall) or np.isnan(original_recall)) else float("nan")
        n_removed = int(fn_removed.sum())
        n_orig_fn = int(fn_mask.sum())
        pct = n_removed / n_orig_fn * 100 if n_orig_fn > 0 else 0
        label_clean = label.replace("\n", " ")
        print(f"{label_clean:<28} {adj_iou:>8.3f} {iou_gain:>+8.3f} {adj_recall:>8.3f} {rec_gain:>+8.3f} {n_removed:>6}/{n_orig_fn:<6} {pct:>5.1f}%")
    print(f"{'Original':<28} {original_iou:>8.3f} {'+0.000':>8} {original_recall:>8.3f} {'+0.000':>8}")


# ──────────────────────────────────────────────
# Dataset-wide metrics
# ──────────────────────────────────────────────

def compute_dataset_metrics(tile_ids=None,
                            erosion_kernel=5, min_component_area=50,
                            min_gt_pixels=100, border_px=5, tp_max_dist=5,
                            tiles_dir=None, t2n_dir=None,
                            conf_dir=None, gt_dir=None):
    """Compute per-tile IoU, recall, precision and their filtered versions.

    Parameters
    ----------
    tile_ids : list of str or None
        Tiles to evaluate. If None, uses ALL_TILE_IDS.
    erosion_kernel, min_component_area, min_gt_pixels, border_px, tp_max_dist
        Hyper-parameters for the combined filters.
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths (passed to load_tile_data).

    Returns
    -------
    results : dict with keys:
        'tile_ids'        — list of tile IDs that have content (non-NaN IoU)
        'iou'             — array of original IoU per tile
        'recall'          — array of original recall per tile
        'precision'       — array of original precision per tile
        'filtered_iou'    — array of filtered IoU per tile
        'filtered_recall' — array of filtered recall per tile
        'n_total'         — total number of tiles evaluated
        'n_empty'         — tiles with no sidewalk in either mask (NaN IoU)
        'n_valid'         — tiles with valid metrics
    """
    if tile_ids is None:
        tile_ids = ALL_TILE_IDS

    ids_valid = []
    ious, recalls, precisions = [], [], []
    filt_ious, filt_recalls = [], []

    for tid in tqdm(tile_ids):
        _, _, t2n_mask, gt_mask = load_tile_data(
            tid, tiles_dir=tiles_dir, t2n_dir=t2n_dir,
            conf_dir=conf_dir, gt_dir=gt_dir)

        iou = compute_iou(t2n_mask, gt_mask)
        if np.isnan(iou):
            continue

        rec = compute_recall(t2n_mask, gt_mask)

        tp = np.sum(t2n_mask & gt_mask)
        total_pred = np.sum(t2n_mask)
        prec = tp / total_pred if total_pred > 0 else float("nan")

        # Filtered metrics
        fn_mask = ~t2n_mask & gt_mask
        fp_mask = t2n_mask & ~gt_mask
        tp_mask = t2n_mask & gt_mask

        _, fn_removed = filter_fn_combined(
            fn_mask, gt_mask, tp_mask,
            erosion_kernel=erosion_kernel,
            min_component_area=min_component_area,
            min_gt_pixels=min_gt_pixels,
            border_px=border_px,
            tp_max_dist=tp_max_dist,
        )
        _, fp_removed = filter_fp_combined(
            fp_mask, t2n_mask, tp_mask,
            erosion_kernel=erosion_kernel,
            min_component_area=min_component_area,
            min_t2n_pixels=min_gt_pixels,
            border_px=border_px,
            tp_max_dist=tp_max_dist,
        )

        f_iou = compute_adjusted_iou(t2n_mask, gt_mask, fn_removed, fp_removed)
        f_rec = compute_adjusted_recall(t2n_mask, gt_mask, fn_removed, fp_removed)

        ids_valid.append(tid)
        ious.append(iou)
        recalls.append(rec)
        precisions.append(prec)
        filt_ious.append(f_iou)
        filt_recalls.append(f_rec)

    return {
        'tile_ids': ids_valid,
        'iou': np.array(ious),
        'recall': np.array(recalls),
        'precision': np.array(precisions),
        'filtered_iou': np.array(filt_ious),
        'filtered_recall': np.array(filt_recalls),
        'n_total': len(tile_ids),
        'n_empty': len(tile_ids) - len(ids_valid),
        'n_valid': len(ids_valid),
    }


def print_dataset_summary(results):
    """Print a compact summary of dataset-wide metrics."""
    n_total = results['n_total']
    n_empty = results['n_empty']
    n_valid = results['n_valid']

    print(f"Dataset: {n_total} total tiles")
    print(f"  Empty (no sidewalk in GT or T2N): {n_empty} ({n_empty / n_total * 100:.1f}%)")
    print(f"  With content (valid IoU):         {n_valid} ({n_valid / n_total * 100:.1f}%)")
    print()

    for name, arr in [
        ("IoU",             results['iou']),
        ("Recall",          results['recall']),
        ("Precision",       results['precision']),
        ("Filtered IoU",    results['filtered_iou']),
        ("Filtered Recall", results['filtered_recall']),
    ]:
        valid = arr[~np.isnan(arr)]
        print(f"  {name:<18}  mean={np.mean(valid):.3f}  median={np.median(valid):.3f}  "
              f"std={np.std(valid):.3f}  min={np.min(valid):.3f}  max={np.max(valid):.3f}")

    # Gains
    iou_gain = results['filtered_iou'] - results['iou']
    rec_gain = results['filtered_recall'] - results['recall']
    valid_iou_g = iou_gain[~np.isnan(iou_gain)]
    valid_rec_g = rec_gain[~np.isnan(rec_gain)]
    print()
    print(f"  IoU gain from filtering:    mean={np.mean(valid_iou_g):+.3f}  median={np.median(valid_iou_g):+.3f}")
    print(f"  Recall gain from filtering: mean={np.mean(valid_rec_g):+.3f}  median={np.median(valid_rec_g):+.3f}")
