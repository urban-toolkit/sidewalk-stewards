"""
Tile2Net Fix Prediction — training utilities.

Dataset class, model, losses, augmentation, training loop,
visualization helpers, and evaluation infrastructure.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from tile2net_gt_utils import (
    ALL_TILE_IDS,
    load_tile_data,
    compute_iou, compute_recall,
    compute_adjusted_iou, compute_adjusted_recall,
    compute_dataset_metrics,
    filter_fn_combined, filter_fp_combined,
    _fmt_metric,
)


# ──────────────────────────────────────────────
# Tile filtering
# ──────────────────────────────────────────────

def get_non_empty_tile_ids(tile_ids=None,
                           tiles_dir=None, t2n_dir=None,
                           conf_dir=None, gt_dir=None):
    """Return tile IDs where at least one of GT or T2N has sidewalk pixels.

    A tile is "empty" if both gt_mask.sum() == 0 and t2n_mask.sum() == 0,
    meaning there is nothing to learn from.  These correspond to NaN IoU
    in the dataset-wide metrics.

    Parameters
    ----------
    tile_ids : list or None
        Tile IDs to check.  Defaults to ALL_TILE_IDS.
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths (passed to load_tile_data).

    Returns
    -------
    list of str
        Tile IDs that have at least some sidewalk content.
    """
    if tile_ids is None:
        tile_ids = ALL_TILE_IDS

    non_empty = []
    for tid in tile_ids:
        _, _, t2n_mask, gt_mask = load_tile_data(
            tid, tiles_dir=tiles_dir, t2n_dir=t2n_dir,
            conf_dir=conf_dir, gt_dir=gt_dir)
        if gt_mask.sum() > 0 or t2n_mask.sum() > 0:
            non_empty.append(tid)
    return non_empty


# ──────────────────────────────────────────────
# Refined metrics for any prediction mask
# ──────────────────────────────────────────────

def compute_refined_metrics(pred_mask, gt_mask,
                            erosion_kernel=5, min_component_area=50,
                            min_gt_pixels=100, border_px=5, tp_max_dist=5):
    """Compute refined (filtered) IoU and recall for any binary prediction.

    Runs the full FN/FP filtering pipeline on the prediction's errors
    against the raw ground truth, then computes adjusted IoU and recall.
    This is the proper way to evaluate any prediction — whether it's
    T2N, a model output, or a corrected mask.

    Parameters
    ----------
    pred_mask : (H, W) bool or uint8
        Binary prediction mask (True/1 = sidewalk).
    gt_mask : (H, W) bool
        Raw ground truth mask (True = sidewalk).
    erosion_kernel, min_component_area, min_gt_pixels, border_px, tp_max_dist
        Filter hyper-parameters (same as SidewalkFixDataset).

    Returns
    -------
    refined_iou : float (or NaN)
    refined_recall : float (or NaN)
    fn_removed : (H, W) bool — FN pixels deemed irrelevant
    fp_removed : (H, W) bool — FP pixels deemed irrelevant
    """
    pred_b = np.asarray(pred_mask, dtype=bool)
    gt_b = np.asarray(gt_mask, dtype=bool)

    fn_mask = ~pred_b & gt_b
    fp_mask = pred_b & ~gt_b
    tp_mask = pred_b & gt_b

    filter_kw = dict(
        erosion_kernel=erosion_kernel,
        min_component_area=min_component_area,
        border_px=border_px,
        tp_max_dist=tp_max_dist,
    )

    _, fn_removed = filter_fn_combined(
        fn_mask, gt_b, tp_mask,
        min_gt_pixels=min_gt_pixels,
        **filter_kw,
    )
    _, fp_removed = filter_fp_combined(
        fp_mask, pred_b, tp_mask,
        min_t2n_pixels=min_gt_pixels,
        **filter_kw,
    )

    r_iou = compute_adjusted_iou(pred_b, gt_b, fn_removed, fp_removed)
    r_rec = compute_adjusted_recall(pred_b, gt_b, fn_removed, fp_removed)

    return r_iou, r_rec, fn_removed, fp_removed


def fmt_metric_delta(val, baseline):
    """Format a metric value with delta from baseline.

    Examples:  '0.45 (+0.12)'   '0.30 (−0.05)'   'NaN'
    """
    if np.isnan(val) or np.isnan(baseline):
        s = "NaN" if np.isnan(val) else f"{val:.2f}"
        return s
    delta = val - baseline
    sign = "+" if delta >= 0 else ""
    return f"{val:.2f} ({sign}{delta:.2f})"


# ──────────────────────────────────────────────
# Stratified train / val split
# ──────────────────────────────────────────────

def stratified_split(tile_ids, val_frac=0.2, seed=42,
                     erosion_kernel=5, min_component_area=50,
                     min_gt_pixels=100, border_px=5, tp_max_dist=5,
                     tiles_dir=None, t2n_dir=None,
                     conf_dir=None, gt_dir=None):
    """Split tile IDs into train/val, stratified by filtered recall buckets.

    Tiles with NaN filtered recall (all GT deemed irrelevant by filters)
    are excluded — nothing meaningful to learn from.

    Buckets: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
    Within each bucket, randomly assign val_frac to val, rest to train.

    Parameters
    ----------
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths (passed through to load_tile_data).

    Returns
    -------
    train_ids, val_ids : list, list
    bucket_info : dict   — per-bucket counts for verification
    n_dropped : int      — tiles excluded due to NaN filtered recall
    """
    results = compute_dataset_metrics(
        tile_ids=tile_ids,
        erosion_kernel=erosion_kernel,
        min_component_area=min_component_area,
        min_gt_pixels=min_gt_pixels,
        border_px=border_px,
        tp_max_dist=tp_max_dist,
        tiles_dir=tiles_dir, t2n_dir=t2n_dir,
        conf_dir=conf_dir, gt_dir=gt_dir,
    )

    ids_arr = np.array(results["tile_ids"])
    recall_arr = np.array(results["filtered_recall"])

    # Drop tiles with NaN filtered recall
    valid = ~np.isnan(recall_arr)
    n_dropped = int((~valid).sum())
    ids_arr = ids_arr[valid]
    recall_arr = recall_arr[valid]

    # Define buckets
    bucket_edges = [0.0, 0.25, 0.5, 0.75, 1.01]
    bucket_labels = ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1.0]"]

    rng = np.random.RandomState(seed)
    train_ids, val_ids = [], []
    bucket_info = {}

    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (recall_arr >= lo) & (recall_arr < hi)
        bucket_ids = ids_arr[mask]

        n = len(bucket_ids)
        n_val = max(1, int(round(n * val_frac)))

        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        val_ids.extend(bucket_ids[val_idx].tolist())
        train_ids.extend(bucket_ids[train_idx].tolist())

        bucket_info[bucket_labels[i]] = {
            "total": n, "train": len(train_idx), "val": len(val_idx)
        }

    return train_ids, val_ids, bucket_info, n_dropped


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SidewalkFixDataset(Dataset):
    """Dataset for sidewalk fix prediction.

    Each sample returns:
        input_tensor : (5, 256, 256) float32
            ch 0-2: RGB satellite (normalized to [0, 1])
            ch 3:   Tile2Net mask (0 or 1)
            ch 4:   Tile2Net confidence (0 to 1)

        target_full : (1, 256, 256) float32
            When use_raw_gt=True:  raw ground truth mask.
            When use_raw_gt=False: filtered GT (GT with irrelevant FN removed).

        target_fixes : (1, 256, 256) float32
            When use_raw_gt=True:  raw FN mask (GT minus T2N, no filtering).
            When use_raw_gt=False: surviving FN mask (filtered fixes).

        tile_id : str
    """

    def __init__(self, tile_ids, use_raw_gt=True,
                 erosion_kernel=5, min_component_area=50,
                 min_gt_pixels=100, border_px=5, tp_max_dist=5,
                 zero_channels=None,
                 tiles_dir=None, t2n_dir=None, conf_dir=None, gt_dir=None):
        self.tile_ids = list(tile_ids)
        self.use_raw_gt = use_raw_gt
        # Channel ablation: list of channel indices to zero out (e.g. [0,1,2] for no RGB)
        self.zero_channels = zero_channels or []
        # Folder paths (None → defaults from tile2net_gt_utils)
        self.tiles_dir = tiles_dir
        self.t2n_dir = t2n_dir
        self.conf_dir = conf_dir
        self.gt_dir = gt_dir
        # Filter hyper-parameters
        self.erosion_kernel = erosion_kernel
        self.min_component_area = min_component_area
        self.min_gt_pixels = min_gt_pixels
        self.border_px = border_px
        self.tp_max_dist = tp_max_dist

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        satellite, confidence, t2n_mask, gt_mask = load_tile_data(
            tile_id, tiles_dir=self.tiles_dir, t2n_dir=self.t2n_dir,
            conf_dir=self.conf_dir, gt_dir=self.gt_dir)

        # --- Input: 5 channels ---
        rgb = satellite.astype(np.float32) / 255.0          # (256,256,3) -> [0,1]
        t2n_ch = t2n_mask.astype(np.float32)                # (256,256) -> {0,1}
        conf_ch = confidence                                 # (256,256) -> [0,1]

        # Stack: (256,256,5) -> (5,256,256)
        input_np = np.concatenate([
            rgb,
            t2n_ch[:, :, None],
            conf_ch[:, :, None],
        ], axis=2).transpose(2, 0, 1)

        if self.use_raw_gt:
            # Raw GT targets — no filtering
            target_full = gt_mask.astype(np.float32)[None, :, :]
            raw_fn = (~t2n_mask & gt_mask).astype(np.float32)
            target_fixes = raw_fn[None, :, :]
        else:
            # Filtered targets
            fn_mask = ~t2n_mask & gt_mask
            fp_mask = t2n_mask & ~gt_mask
            tp_mask = t2n_mask & gt_mask

            surviving_fn, fn_removed = filter_fn_combined(
                fn_mask, gt_mask, tp_mask,
                erosion_kernel=self.erosion_kernel,
                min_component_area=self.min_component_area,
                min_gt_pixels=self.min_gt_pixels,
                border_px=self.border_px,
                tp_max_dist=self.tp_max_dist,
            )
            _, fp_removed = filter_fp_combined(
                fp_mask, t2n_mask, tp_mask,
                erosion_kernel=self.erosion_kernel,
                min_component_area=self.min_component_area,
                min_t2n_pixels=self.min_gt_pixels,
                border_px=self.border_px,
                tp_max_dist=self.tp_max_dist,
            )

            filtered_gt = gt_mask & ~fn_removed
            target_full = filtered_gt.astype(np.float32)[None, :, :]
            target_fixes = surviving_fn.astype(np.float32)[None, :, :]

        # Channel ablation: zero out specified channels
        if self.zero_channels:
            for ch in self.zero_channels:
                input_np[ch] = 0.0

        return (
            torch.from_numpy(input_np),
            torch.from_numpy(target_full),
            torch.from_numpy(target_fixes),
            tile_id,
        )


# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────

def augment_sample(inp, tgt_full, tgt_fixes, crop_size=224):
    """Apply augmentations to a single training sample.

    Spatial augmentations (flip, rot, crop) are applied jointly to all tensors.
    Color jitter + noise are applied to RGB channels (0:3) of inp only.

    Parameters
    ----------
    inp        : (5, H, W) float32
    tgt_full   : (1, H, W) float32
    tgt_fixes  : (1, H, W) float32
    crop_size  : int — size of random crop (default 224)

    Returns
    -------
    inp, tgt_full, tgt_fixes : augmented tensors (crop_size × crop_size)
    """
    # --- 1. Random horizontal flip ---
    if torch.rand(1).item() > 0.5:
        inp = inp.flip(-1)
        tgt_full = tgt_full.flip(-1)
        tgt_fixes = tgt_fixes.flip(-1)

    # --- 2. Random vertical flip ---
    if torch.rand(1).item() > 0.5:
        inp = inp.flip(-2)
        tgt_full = tgt_full.flip(-2)
        tgt_fixes = tgt_fixes.flip(-2)

    # --- 3. Random 90° rotation (0, 1, 2, or 3 times) ---
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        inp = torch.rot90(inp, k, dims=(-2, -1))
        tgt_full = torch.rot90(tgt_full, k, dims=(-2, -1))
        tgt_fixes = torch.rot90(tgt_fixes, k, dims=(-2, -1))

    # --- 4. Random crop ---
    _, h, w = inp.shape
    if crop_size < h and crop_size < w:
        top = torch.randint(0, h - crop_size, (1,)).item()
        left = torch.randint(0, w - crop_size, (1,)).item()
        inp = inp[:, top:top+crop_size, left:left+crop_size]
        tgt_full = tgt_full[:, top:top+crop_size, left:left+crop_size]
        tgt_fixes = tgt_fixes[:, top:top+crop_size, left:left+crop_size]

    # --- 5. Color jitter (RGB only) ---
    rgb = inp[:3]  # (3, H, W)
    # Brightness
    rgb = rgb + (torch.rand(1).item() - 0.5) * 0.2
    # Contrast
    mean = rgb.mean()
    rgb = (rgb - mean) * (0.8 + torch.rand(1).item() * 0.4) + mean
    # Saturation (simple: blend toward grayscale)
    gray = rgb.mean(dim=0, keepdim=True)
    sat_factor = 0.7 + torch.rand(1).item() * 0.6
    rgb = gray + (rgb - gray) * sat_factor
    rgb = rgb.clamp(0, 1)

    # --- 6. Gaussian noise (RGB only) ---
    rgb = rgb + torch.randn_like(rgb) * 0.02
    rgb = rgb.clamp(0, 1)

    inp = torch.cat([rgb, inp[3:]], dim=0)  # re-attach T2N + confidence

    return inp, tgt_full, tgt_fixes


class AugmentedDataset(Dataset):
    """Wraps SidewalkFixDataset with on-the-fly augmentation."""

    def __init__(self, base_dataset, crop_size=224, augment=True):
        self.base = base_dataset
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        inp, tgt_full, tgt_fixes, tile_id = self.base[idx]
        if self.augment:
            inp, tgt_full, tgt_fixes = augment_sample(
                inp, tgt_full, tgt_fixes, crop_size=self.crop_size
            )
        return inp, tgt_full, tgt_fixes, tile_id


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class ResidualFixNet(nn.Module):
    """Pretrained encoder U-Net with residual connection and dual/triple heads.

    - Frozen ResNet34 encoder (pretrained on ImageNet)
    - 5-channel input: RGB (pretrained) + T2N mask + confidence (zero-init)
    - Trainable decoder + dual 1×1 heads (+ optional remove head)
    - head_full: when warm_start_t2n=True → residual σ(logit(T2N) + Δ)
                 when warm_start_t2n=False → direct σ(logits)
    - head_fix:  direct → σ(logits) — predicts pixels to ADD
    - head_remove (optional): direct → σ(logits) — predicts pixels to REMOVE

    Parameters
    ----------
    warm_start_t2n : bool (default False)
        If True, full head uses residual on T2N: σ(logit(T2N) + Δ).
        If False, full head predicts GT directly: σ(logits).
    enable_remove : bool (default False)
        If True, adds a third head (head_remove) for predicting pixel removal.
        forward() returns 3 outputs: (out_full, out_fix, out_remove).
        If False, forward() returns 2 outputs as before: (out_full, out_fix).
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 freeze_encoder=True, warm_start_t2n=False, eps=1e-6,
                 enable_remove=False):
        super().__init__()
        self.eps = eps
        self.warm_start_t2n = warm_start_t2n
        self.enable_remove = enable_remove

        # Build a U-Net with 2 output classes (we'll replace the head)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,          # start with 3, we'll patch to 5
            classes=1,              # placeholder, we replace heads below
            activation=None,        # raw logits
        )

        # --- Patch first conv: 3ch → 5ch ---
        old_conv = self.unet.encoder.conv1
        new_conv = nn.Conv2d(
            5, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            # Copy pretrained RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Zero-init the 2 extra channels (T2N mask + confidence)
            new_conv.weight[:, 3:, :, :] = 0.0
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.unet.encoder.conv1 = new_conv

        # --- Freeze encoder ---
        if freeze_encoder:
            for param in self.unet.encoder.parameters():
                param.requires_grad = False
            # Unfreeze the patched first conv (the 2 extra channels need to learn)
            for param in self.unet.encoder.conv1.parameters():
                param.requires_grad = True

        # --- Dual heads: replace the single segmentation head ---
        decoder_out_ch = self.unet.segmentation_head[0].in_channels

        self.head_full = nn.Conv2d(decoder_out_ch, 1, 1)  # full prediction
        self.head_fix  = nn.Conv2d(decoder_out_ch, 1, 1)  # fix mask (add pixels)

        # Optional remove head
        if self.enable_remove:
            self.head_remove = nn.Conv2d(decoder_out_ch, 1, 1)  # remove mask

        # Zero-init head_full when using warm start so initial delta ≈ 0 → output ≈ T2N
        if self.warm_start_t2n:
            nn.init.zeros_(self.head_full.weight)
            nn.init.zeros_(self.head_full.bias)

        # Remove SMP's default segmentation head (we use our own)
        self.unet.segmentation_head = nn.Identity()

    def _safe_logit(self, p):
        p = p.clamp(self.eps, 1.0 - self.eps)
        return torch.log(p / (1.0 - p))

    def forward(self, x):
        """
        x : (B, 5, H, W)  — ch 0-2: RGB, ch 3: T2N mask, ch 4: confidence

        Returns
        -------
        If enable_remove=False (default):
            out_full : (B, 1, H, W)  — corrected segmentation (sigmoid)
            out_fix  : (B, 1, H, W)  — fix-only mask (sigmoid)
        If enable_remove=True:
            out_full   : (B, 1, H, W)
            out_fix    : (B, 1, H, W)  — pixels to ADD
            out_remove : (B, 1, H, W)  — pixels to REMOVE
        """
        # Forward through encoder + decoder (skip SMP's head)
        features = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)  # (B, decoder_out_ch, H, W)

        # Full head
        logits_full = self.head_full(decoder_out)
        if self.warm_start_t2n:
            t2n_prob = x[:, 3:4, :, :]
            t2n_logit = self._safe_logit(t2n_prob)
            out_full = torch.sigmoid(t2n_logit + logits_full)
        else:
            out_full = torch.sigmoid(logits_full)

        # Fix head (always direct) — predicts pixels to add
        out_fix = torch.sigmoid(self.head_fix(decoder_out))

        if self.enable_remove:
            out_remove = torch.sigmoid(self.head_remove(decoder_out))
            return out_full, out_fix, out_remove

        return out_full, out_fix


# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0):
    """Tversky loss — asymmetric Dice that penalizes FN more (beta > alpha).

    pred, target : (B, 1, H, W) float in [0, 1]
    alpha : weight on FP
    beta  : weight on FN  (beta > alpha → recall-focused)
    """
    pred_flat = pred.reshape(-1)
    tgt_flat = target.reshape(-1)

    tp = (pred_flat * tgt_flat).sum()
    fp = (pred_flat * (1 - tgt_flat)).sum()
    fn = ((1 - pred_flat) * tgt_flat).sum()

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky


def focal_loss(pred, target, gamma=2.0, alpha=0.75):
    """Focal loss for sparse binary targets.

    pred, target : (B, 1, H, W) float in [0, 1]
    gamma : focusing parameter (higher → more focus on hard examples)
    alpha : weight for positive class (fixes are rare → alpha > 0.5)
    """
    pred = pred.clamp(1e-6, 1 - 1e-6)

    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    pt = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - pt) ** gamma

    alpha_weight = target * alpha + (1 - target) * (1 - alpha)

    return (alpha_weight * focal_weight * bce).mean()


def combined_loss(pred_full, pred_fix, tgt_full, tgt_fix,
                  pred_remove=None, tgt_remove=None,
                  tversky_alpha=0.3, tversky_beta=0.7,
                  focal_gamma=2.0, focal_alpha=0.75,
                  w_full=1.0, w_fix=0.5, w_remove=0.5):
    """Combined loss for dual/triple-head model.

    w_full, w_fix, w_remove : relative weights of the heads.
    pred_remove, tgt_remove : optional — only used when enable_remove=True.
    """
    loss_full = tversky_loss(pred_full, tgt_full,
                             alpha=tversky_alpha, beta=tversky_beta)
    loss_fix = focal_loss(pred_fix, tgt_fix,
                          gamma=focal_gamma, alpha=focal_alpha)
    total = w_full * loss_full + w_fix * loss_fix

    l_remove = 0.0
    if pred_remove is not None and tgt_remove is not None:
        loss_remove = focal_loss(pred_remove, tgt_remove,
                                 gamma=focal_gamma, alpha=focal_alpha)
        total = total + w_remove * loss_remove
        l_remove = loss_remove.item()

    return total, loss_full.item(), loss_fix.item(), l_remove


# ──────────────────────────────────────────────
# Training sample visualization
# ──────────────────────────────────────────────

def display_training_sample(tile_id=None, dataset=None,
                            erosion_kernel=5, min_component_area=50,
                            min_gt_pixels=100, border_px=5, tp_max_dist=5,
                            figscale=4.0,
                            tiles_dir=None, t2n_dir=None,
                            conf_dir=None, gt_dir=None):
    """Visualize a single training sample: inputs, targets, and residual.

    Shows a 2×4 grid:
        Row 1: RGB satellite | T2N mask | T2N confidence | (empty)
        Row 2: Filtered GT | Fixes only | Filtered T2N | Residual

    The suptitle includes IoU, filtered IoU, recall, and filtered recall.

    Parameters
    ----------
    tile_id : str or None
        Tile to display. If None, picks a random tile.
    dataset : SidewalkFixDataset or None
        If provided, loads the sample from the dataset (uses its filter params).
        If None, creates a temporary one with the given filter hyper-parameters.
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths (passed to load_tile_data / SidewalkFixDataset).
    """
    filter_kwargs = dict(
        erosion_kernel=erosion_kernel,
        min_component_area=min_component_area,
        min_gt_pixels=min_gt_pixels,
        border_px=border_px,
        tp_max_dist=tp_max_dist,
    )

    if tile_id is None:
        if dataset is not None:
            tile_id = dataset.tile_ids[np.random.randint(len(dataset))]
        else:
            tile_id = np.random.choice(ALL_TILE_IDS)

    # Load via dataset if available (uses its filter params), otherwise manually
    if dataset is not None and tile_id in dataset.tile_ids:
        idx = dataset.tile_ids.index(tile_id)
        inp, tgt_full, tgt_fixes, tid = dataset[idx]
        filter_kwargs = dict(
            erosion_kernel=dataset.erosion_kernel,
            min_component_area=dataset.min_component_area,
            min_gt_pixels=dataset.min_gt_pixels,
            border_px=dataset.border_px,
            tp_max_dist=dataset.tp_max_dist,
        )
    else:
        ds_tmp = SidewalkFixDataset([tile_id], use_raw_gt=False,
                                    tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                                    conf_dir=conf_dir, gt_dir=gt_dir,
                                    **filter_kwargs)
        inp, tgt_full, tgt_fixes, tid = ds_tmp[0]

    inp_np = inp.numpy()

    # Compute metrics — reload raw masks
    satellite, confidence, t2n_mask, gt_mask = load_tile_data(
        tile_id, tiles_dir=tiles_dir, t2n_dir=t2n_dir,
        conf_dir=conf_dir, gt_dir=gt_dir)

    iou = compute_iou(t2n_mask, gt_mask)
    rec = compute_recall(t2n_mask, gt_mask)

    fn_mask = ~t2n_mask & gt_mask
    fp_mask = t2n_mask & ~gt_mask
    tp_mask = t2n_mask & gt_mask

    _, fn_removed = filter_fn_combined(fn_mask, gt_mask, tp_mask, **filter_kwargs)
    _, fp_removed = filter_fp_combined(
        fp_mask, t2n_mask, tp_mask,
        erosion_kernel=filter_kwargs['erosion_kernel'],
        min_component_area=filter_kwargs['min_component_area'],
        min_t2n_pixels=filter_kwargs['min_gt_pixels'],
        border_px=filter_kwargs['border_px'],
        tp_max_dist=filter_kwargs['tp_max_dist'],
    )

    f_iou = compute_adjusted_iou(t2n_mask, gt_mask, fn_removed, fp_removed)
    f_rec = compute_adjusted_recall(t2n_mask, gt_mask, fn_removed, fp_removed)

    filtered_gt = (gt_mask & ~fn_removed).astype(np.float32)
    filtered_t2n = (t2n_mask & ~fp_removed).astype(np.float32)

    # --- Plot ---
    fig, axes = plt.subplots(2, 4, figsize=(figscale * 4, figscale * 2))

    # Row 1: input channels
    axes[0, 0].imshow(inp_np[:3].transpose(1, 2, 0))
    axes[0, 0].set_title("Input: RGB satellite")

    axes[0, 1].imshow(inp_np[3], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Input: T2N mask")

    axes[0, 2].imshow(inp_np[4], cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title("Input: T2N confidence")

    axes[0, 3].axis("off")

    # Row 2: targets
    axes[1, 0].imshow(tgt_full[0], cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Target: Filtered GT (full)")

    axes[1, 1].imshow(tgt_fixes[0], cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Target: Fixes only (surv. FN)")

    axes[1, 2].imshow(filtered_t2n, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Filtered T2N (FP removed)")

    # Residual: filtered_gt - filtered_t2n
    residual = filtered_gt - filtered_t2n
    axes[1, 3].imshow(residual, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 3].set_title("Residual (filt. GT - filt. T2N)")
    n_add = int((residual > 0.5).sum())
    n_remove = int((residual < -0.5).sum())
    axes[1, 3].set_xlabel(f"Add: {n_add}  Remove: {n_remove}", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"{tile_id}\n"
        f"IoU: {_fmt_metric(iou)}  Filt. IoU: {_fmt_metric(f_iou)}  |  "
        f"Recall: {_fmt_metric(rec)}  Filt. Recall: {_fmt_metric(f_rec)}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Prediction visualization (post-training)
# ──────────────────────────────────────────────

def display_predictions(model, tile_ids, device, title="Model Predictions",
                        subtitles=None, figscale=3.5,
                        zero_channels=None,
                        tiles_dir=None, t2n_dir=None,
                        conf_dir=None, gt_dir=None):
    """Display 7-column evaluation grid for a list of tiles.

    Columns: Satellite | T2N Confidence | T2N Input | Ground Truth |
             Pred Full | Fix Target | Pred Fix

    Metrics shown under T2N Input, Pred Full, and Pred Fix:
        Line 1: rIoU (refined) and IoU (raw), with deltas for predictions
        Line 2: rRec (refined) and Rec (raw), with deltas for predictions

    Parameters
    ----------
    model      : nn.Module — must return (out_full, out_fix) in [0, 1]
    tile_ids   : list of str — tiles to evaluate
    device     : torch.device
    title      : str — figure suptitle
    subtitles  : list of str or None — extra label per row (e.g. bucket name).
                 If provided, shown below the tile ID on the y-axis.
    figscale   : float — size multiplier per cell
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths (passed to SidewalkFixDataset / load_tile_data).
    """
    model.eval()
    eval_ds = SidewalkFixDataset(tile_ids, zero_channels=zero_channels,
                                 tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                                 conf_dir=conf_dir, gt_dir=gt_dir)

    n_rows = len(tile_ids)
    n_cols = 7
    col_titles = ["Satellite", "T2N Confidence", "T2N Input", "Ground Truth",
                  "Pred Full", "Fix Target", "Pred Fix"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * figscale, figscale * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, tid in enumerate(tile_ids):
        inp, tgt_full, tgt_fixes, _ = eval_ds[i]

        with torch.no_grad():
            outputs = model(inp.unsqueeze(0).to(device))
        has_remove = hasattr(model, 'enable_remove') and model.enable_remove
        if has_remove:
            pred_full, pred_fix, pred_remove = outputs
            pred_remove_np = pred_remove[0, 0].cpu().numpy()
        else:
            pred_full, pred_fix = outputs
            pred_remove_np = None
        pred_full_np = pred_full[0, 0].cpu().numpy()
        pred_fix_np  = pred_fix[0, 0].cpu().numpy()

        inp_np = inp.numpy()
        t2n_binary = (inp_np[3] > 0.5).astype(bool)

        _, _, _, gt_mask = load_tile_data(tid, tiles_dir=tiles_dir,
                                          t2n_dir=t2n_dir, conf_dir=conf_dir,
                                          gt_dir=gt_dir)

        t2n_iou  = compute_iou(t2n_binary, gt_mask)
        t2n_rec  = compute_recall(t2n_binary, gt_mask)
        t2n_riou, t2n_rrec, _, _ = compute_refined_metrics(t2n_binary, gt_mask)

        pf_bin = pred_full_np > 0.5
        pf_iou  = compute_iou(pf_bin, gt_mask)
        pf_rec  = compute_recall(pf_bin, gt_mask)
        pf_riou, pf_rrec, _, _ = compute_refined_metrics(pf_bin, gt_mask)

        if pred_remove_np is not None:
            px_bin = (t2n_binary & ~(pred_remove_np > 0.5)) | (pred_fix_np > 0.5)
        else:
            px_bin = t2n_binary | (pred_fix_np > 0.5)
        px_iou  = compute_iou(px_bin, gt_mask)
        px_rec  = compute_recall(px_bin, gt_mask)
        px_riou, px_rrec, _, _ = compute_refined_metrics(px_bin, gt_mask)

        ylabel = tid if subtitles is None else f"{tid}\n{subtitles[i]}"
        axes[i, 0].set_ylabel(ylabel, fontsize=8, rotation=90,
                               labelpad=10, va="center")

        axes[i, 0].imshow(inp_np[:3].transpose(1, 2, 0))
        axes[i, 1].imshow(inp_np[4], cmap="hot", vmin=0, vmax=1)

        axes[i, 2].imshow(inp_np[3], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_xlabel(
            f"rIoU={t2n_riou:.2f}  IoU={t2n_iou:.2f}\n"
            f"rRec={t2n_rrec:.2f}  Rec={t2n_rec:.2f}", fontsize=7)

        axes[i, 3].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)

        axes[i, 4].imshow(pred_full_np, cmap="gray", vmin=0, vmax=1)
        axes[i, 4].set_xlabel(
            f"rIoU={fmt_metric_delta(pf_riou, t2n_riou)}  "
            f"IoU={fmt_metric_delta(pf_iou, t2n_iou)}\n"
            f"rRec={fmt_metric_delta(pf_rrec, t2n_rrec)}  "
            f"Rec={fmt_metric_delta(pf_rec, t2n_rec)}", fontsize=7)

        axes[i, 5].imshow(tgt_fixes[0].numpy(), cmap="gray", vmin=0, vmax=1)

        # Pred Fix overlay: T2N = white, new fix pixels = pastel green
        overlay = np.zeros((256, 256, 3), dtype=np.uint8)
        overlay[t2n_binary] = [255, 255, 255]
        new_only = (pred_fix_np > 0.5) & ~t2n_binary
        overlay[new_only] = [193, 225, 193]  # #C1E1C1
        axes[i, 6].imshow(overlay)
        axes[i, 6].set_xlabel(
            f"rIoU={fmt_metric_delta(px_riou, t2n_riou)}  "
            f"IoU={fmt_metric_delta(px_iou, t2n_iou)}\n"
            f"rRec={fmt_metric_delta(px_rrec, t2n_rrec)}  "
            f"Rec={fmt_metric_delta(px_rec, t2n_rec)}", fontsize=7)

    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def evaluate_by_bucket(model, device, split, train_ids, val_ids, bucket_info,
                       n_per_bucket=1, seed=None, figscale=3.5,
                       tiles_dir=None, t2n_dir=None,
                       conf_dir=None, gt_dir=None):
    """Sample random tiles from each recall bucket and display predictions.

    Parameters
    ----------
    model        : nn.Module
    device       : torch.device
    split        : str — "train" or "val"
    train_ids    : list — ordered by bucket (output of stratified_split)
    val_ids      : list — ordered by bucket (output of stratified_split)
    bucket_info  : dict — per-bucket counts (output of stratified_split)
    n_per_bucket : int — number of random tiles per bucket (default 1)
    seed         : int or None — random seed for reproducibility
    figscale     : float — size multiplier per cell
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override default folder paths.
    """
    ids_list = train_ids if split == "train" else val_ids
    count_key = "train" if split == "train" else "val"

    bucket_labels = list(bucket_info.keys())
    bucket_sizes = [bucket_info[b][count_key] for b in bucket_labels]
    bucket_starts = [0] + list(np.cumsum(bucket_sizes[:-1]))

    rng = np.random.RandomState(seed)
    sample_ids = []
    sample_buckets = []

    for label, start, size in zip(bucket_labels, bucket_starts, bucket_sizes):
        n = min(n_per_bucket, size)
        idxs = rng.choice(size, n, replace=False)
        for idx in idxs:
            sample_ids.append(ids_list[start + idx])
            sample_buckets.append(label)

    print(f"{split.capitalize()} samples ({n_per_bucket} per bucket, "
          f"{len(sample_ids)} total):")
    for tid, bucket in zip(sample_ids, sample_buckets):
        _, _, t2n, gt = load_tile_data(tid, tiles_dir=tiles_dir,
                                       t2n_dir=t2n_dir, conf_dir=conf_dir,
                                       gt_dir=gt_dir)
        _, rrec, _, _ = compute_refined_metrics(t2n, gt)
        rec = compute_recall(t2n, gt)
        print(f"  {bucket:<16}  {tid}  rRec={rrec:.3f}  Rec={rec:.3f}")

    split_label = "Training Set" if split == "train" else "Validation Set (UNSEEN)"
    title = f"{split_label} — {n_per_bucket} per bucket"

    display_predictions(model, sample_ids, device, title=title,
                        subtitles=sample_buckets, figscale=figscale,
                        tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                        conf_dir=conf_dir, gt_dir=gt_dir)


# ──────────────────────────────────────────────
# Evaluation helpers (metrics over tile sets)
# ──────────────────────────────────────────────

def evaluate_tiles(model, tile_ids, device,
                   zero_channels=None,
                   tiles_dir=None, t2n_dir=None, conf_dir=None, gt_dir=None):
    """Compute mean refined metrics over a set of tiles (full 256×256, no aug).

    Runs the model on each tile, thresholds predictions, computes refined
    IoU/recall for both heads and the T2N baseline.
    If the model has enable_remove=True, also evaluates the combined
    add+remove correction: (t2n & ~remove) | add.

    Parameters
    ----------
    model    : nn.Module — returns (out_full, out_fix) or (out_full, out_fix, out_remove)
    tile_ids : list of str
    device   : torch.device
    zero_channels : list of int or None
        Channel indices to zero out for ablation (must match training).
    tiles_dir, t2n_dir, conf_dir, gt_dir : str or None
        Override data folders (default: standardized dorchester_exports).

    Returns
    -------
    dict with keys:
        't2n_riou', 't2n_rrec', 't2n_iou', 't2n_rec'  — T2N baseline
        'full_riou', 'full_rrec', 'full_iou', 'full_rec' — pred_full head
        'fix_riou', 'fix_rrec', 'fix_iou', 'fix_rec'   — pred_fix (T2N | fix)
    All values are floats (NaN tiles excluded from mean).
    """
    model.eval()
    has_remove = hasattr(model, 'enable_remove') and model.enable_remove
    ds = SidewalkFixDataset(tile_ids, zero_channels=zero_channels,
                            tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                            conf_dir=conf_dir, gt_dir=gt_dir)

    keys = ['t2n_riou', 't2n_rrec', 't2n_iou', 't2n_rec',
            'full_riou', 'full_rrec', 'full_iou', 'full_rec',
            'fix_riou', 'fix_rrec', 'fix_iou', 'fix_rec']
    accum = {k: [] for k in keys}

    for i in range(len(ds)):
        inp, _, _, tid = ds[i]

        with torch.no_grad():
            outputs = model(inp.unsqueeze(0).to(device))

        if has_remove:
            pred_full, pred_fix, pred_remove = outputs
            pred_remove_np = pred_remove[0, 0].cpu().numpy()
        else:
            pred_full, pred_fix = outputs
            pred_remove_np = None

        pred_full_np = pred_full[0, 0].cpu().numpy()
        pred_fix_np  = pred_fix[0, 0].cpu().numpy()

        inp_np = inp.numpy()
        t2n_binary = (inp_np[3] > 0.5).astype(bool)
        _, _, _, gt_mask = load_tile_data(tid, tiles_dir=tiles_dir,
                                              t2n_dir=t2n_dir, conf_dir=conf_dir,
                                              gt_dir=gt_dir)

        # Fix combined mask: remove first, then add (option 2)
        if pred_remove_np is not None:
            fix_bin = (t2n_binary & ~(pred_remove_np > 0.5)) | (pred_fix_np > 0.5)
        else:
            fix_bin = t2n_binary | (pred_fix_np > 0.5)

        # Refined metrics
        t2n_riou, t2n_rrec, _, _ = compute_refined_metrics(t2n_binary, gt_mask)
        pf_riou, pf_rrec, _, _   = compute_refined_metrics(
            pred_full_np > 0.5, gt_mask)
        px_riou, px_rrec, _, _   = compute_refined_metrics(fix_bin, gt_mask)

        # Raw metrics
        t2n_iou = compute_iou(t2n_binary, gt_mask)
        t2n_rec = compute_recall(t2n_binary, gt_mask)
        pf_iou  = compute_iou(pred_full_np > 0.5, gt_mask)
        pf_rec  = compute_recall(pred_full_np > 0.5, gt_mask)
        px_iou  = compute_iou(fix_bin, gt_mask)
        px_rec  = compute_recall(fix_bin, gt_mask)

        accum['t2n_riou'].append(t2n_riou)
        accum['t2n_rrec'].append(t2n_rrec)
        accum['t2n_iou'].append(t2n_iou)
        accum['t2n_rec'].append(t2n_rec)
        accum['full_riou'].append(pf_riou)
        accum['full_rrec'].append(pf_rrec)
        accum['full_iou'].append(pf_iou)
        accum['full_rec'].append(pf_rec)
        accum['fix_riou'].append(px_riou)
        accum['fix_rrec'].append(px_rrec)
        accum['fix_iou'].append(px_iou)
        accum['fix_rec'].append(px_rec)

    result = {}
    for k in keys:
        vals = [v for v in accum[k] if not np.isnan(v)]
        result[k] = np.mean(vals) if vals else float('nan')
    return result


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_model(model, train_tile_ids, val_ids, bucket_info, device,
                n_epochs=200, batch_size=4, lr=1e-3, weight_decay=1e-5,
                eval_every=10, crop_size=224,
                use_raw_gt=True, warm_start_t2n=False,
                zero_channels=None,
                tiles_dir=None, t2n_dir=None, conf_dir=None, gt_dir=None):
    """Run the training loop with per-epoch refined metrics tracking.

    Parameters
    ----------
    model           : ResidualFixNet — already on device
    train_tile_ids  : list — tile IDs for training (e.g. overfit subset)
    val_ids         : list — full val set IDs (ordered by bucket)
    bucket_info     : dict — per-bucket counts (from stratified_split)
    device          : torch.device
    n_epochs        : int
    batch_size      : int
    lr              : float
    weight_decay    : float
    eval_every      : int — compute refined metrics every N epochs
    crop_size       : int — random crop size for augmentation
    use_raw_gt      : bool (default True)
        If True, train on raw GT. If False, train on filtered GT.
    warm_start_t2n  : bool (default False)
        Should match model.warm_start_t2n. Logged for reference.
    zero_channels   : list of int or None
        Channel indices to zero out for ablation (e.g. [0,1,2] = no RGB).
    tiles_dir       : str or None — override default tiles folder
    t2n_dir         : str or None — override default T2N masks folder
    conf_dir        : str or None — override default confidence masks folder
    gt_dir          : str or None — override default ground truth masks folder

    Returns
    -------
    history : dict — loss and metric curves
    train_baseline : dict — T2N baseline metrics on training tiles
    val_baseline : dict — T2N baseline metrics on sampled val tiles
    val_final : dict — final metrics on full val set
    """
    # Dataset and dataloader
    train_base = SidewalkFixDataset(train_tile_ids, use_raw_gt=use_raw_gt,
                                    zero_channels=zero_channels,
                                    tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                                    conf_dir=conf_dir, gt_dir=gt_dir)
    train_ds = AugmentedDataset(train_base, crop_size=crop_size, augment=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # Val sample: ~5% per bucket (min 1) for fast per-epoch eval
    val_sample_rng = np.random.RandomState(42)
    val_bucket_labels = list(bucket_info.keys())
    val_bucket_sizes = [bucket_info[b]["val"] for b in val_bucket_labels]
    val_bucket_starts = [0] + list(np.cumsum(val_bucket_sizes[:-1]))
    val_sample_ids = []
    for label, start, size in zip(val_bucket_labels, val_bucket_starts,
                                  val_bucket_sizes):
        n_sample = max(1, int(0.05 * size))
        idxs = val_sample_rng.choice(size, n_sample, replace=False)
        for idx in idxs:
            val_sample_ids.append(val_ids[start + idx])

    print(f"Training config: use_raw_gt={use_raw_gt}, "
          f"warm_start_t2n={warm_start_t2n}, "
          f"zero_channels={zero_channels}")
    print(f"Train tiles: {len(train_tile_ids)}, Val sample: {len(val_sample_ids)}")
    print(f"Val sample tiles ({len(val_sample_ids)}):")
    for tid in val_sample_ids:
        _, _, t2n, gt = load_tile_data(tid, tiles_dir=tiles_dir,
                                       t2n_dir=t2n_dir, conf_dir=conf_dir,
                                       gt_dir=gt_dir)
        _, rrec, _, _ = compute_refined_metrics(t2n, gt)
        print(f"  {tid}  rRec={rrec:.3f}")

    # T2N baseline metrics (computed once)
    train_baseline = evaluate_tiles(model, train_tile_ids, device,
                                    zero_channels=zero_channels,
                                    tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                                    conf_dir=conf_dir, gt_dir=gt_dir)
    val_baseline = evaluate_tiles(model, val_sample_ids, device,
                                  zero_channels=zero_channels,
                                  tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                                  conf_dir=conf_dir, gt_dir=gt_dir)
    print(f"\n── T2N Baselines ──")
    print(f"Train ({len(train_tile_ids)} tiles): "
          f"rIoU={train_baseline['t2n_riou']:.4f}  "
          f"rRec={train_baseline['t2n_rrec']:.4f}")
    print(f"Val   ({len(val_sample_ids)} tiles): "
          f"rIoU={val_baseline['t2n_riou']:.4f}  "
          f"rRec={val_baseline['t2n_rrec']:.4f}")

    # Detect if model has remove head
    has_remove = hasattr(model, 'enable_remove') and model.enable_remove
    if has_remove:
        print("Model mode: add + remove (triple head)")
    else:
        print("Model mode: add only (dual head)")

    # Best-model checkpointing (based on val fix_riou)
    import copy
    best_val_riou = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    # History
    history = {
        "loss": [], "loss_full": [], "loss_fix": [], "loss_remove": [],
        "train_full_riou": [], "train_full_rrec": [],
        "train_fix_riou": [], "train_fix_rrec": [],
        "val_full_riou": [], "val_full_rrec": [],
        "val_fix_riou": [], "val_fix_rrec": [],
        "metric_epochs": [],
    }

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss, epoch_full, epoch_fix, epoch_remove, n_batches = 0, 0, 0, 0, 0

        for batch_inp, batch_tgt_full, batch_tgt_fix, _ in train_dl:
            batch_inp = batch_inp.to(device)
            batch_tgt_full = batch_tgt_full.to(device)
            batch_tgt_fix = batch_tgt_fix.to(device)

            # Remove target: pixels in T2N but NOT in GT (pixels to erase)
            pred_remove_out = None
            tgt_remove = None
            if has_remove:
                pred_full, pred_fix, pred_remove_out = model(batch_inp)
                t2n_ch = batch_inp[:, 3:4, :, :]
                tgt_remove = (t2n_ch > 0.5).float() * (1.0 - batch_tgt_full)
            else:
                pred_full, pred_fix = model(batch_inp)

            loss, l_full, l_fix, l_remove = combined_loss(
                pred_full, pred_fix, batch_tgt_full, batch_tgt_fix,
                pred_remove=pred_remove_out, tgt_remove=tgt_remove)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_full += l_full
            epoch_fix += l_fix
            epoch_remove += l_remove
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_full = epoch_full / n_batches
        avg_fix = epoch_fix / n_batches
        avg_remove = epoch_remove / n_batches if n_batches > 0 else 0
        history["loss"].append(avg_loss)
        history["loss_full"].append(avg_full)
        history["loss_fix"].append(avg_fix)
        history["loss_remove"].append(avg_remove)

        # Refined metrics every eval_every epochs + first + last
        do_eval = ((epoch == 0) or ((epoch + 1) % eval_every == 0)
                   or (epoch == n_epochs - 1))
        if do_eval:
            train_m = evaluate_tiles(model, train_tile_ids, device,
                                    zero_channels=zero_channels,
                                    tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                                    conf_dir=conf_dir, gt_dir=gt_dir)
            val_m = evaluate_tiles(model, val_sample_ids, device,
                                   zero_channels=zero_channels,
                                   tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                                   conf_dir=conf_dir, gt_dir=gt_dir)

            history["train_full_riou"].append(train_m['full_riou'])
            history["train_full_rrec"].append(train_m['full_rrec'])
            history["train_fix_riou"].append(train_m['fix_riou'])
            history["train_fix_rrec"].append(train_m['fix_rrec'])
            history["val_full_riou"].append(val_m['full_riou'])
            history["val_full_rrec"].append(val_m['full_rrec'])
            history["val_fix_riou"].append(val_m['fix_riou'])
            history["val_fix_rrec"].append(val_m['fix_rrec'])
            history["metric_epochs"].append(epoch + 1)

            # Best-model checkpoint
            if val_m['fix_riou'] > best_val_riou:
                best_val_riou = val_m['fix_riou']
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

            model.train()

            t_riou_d = fmt_metric_delta(train_m['full_riou'],
                                        train_baseline['t2n_riou'])
            t_rrec_d = fmt_metric_delta(train_m['full_rrec'],
                                        train_baseline['t2n_rrec'])
            tf_riou_d = fmt_metric_delta(train_m['fix_riou'],
                                         train_baseline['t2n_riou'])
            tf_rrec_d = fmt_metric_delta(train_m['fix_rrec'],
                                         train_baseline['t2n_rrec'])
            v_riou_d = fmt_metric_delta(val_m['full_riou'],
                                        val_baseline['t2n_riou'])
            v_rrec_d = fmt_metric_delta(val_m['full_rrec'],
                                        val_baseline['t2n_rrec'])
            vf_riou_d = fmt_metric_delta(val_m['fix_riou'],
                                         val_baseline['t2n_riou'])
            vf_rrec_d = fmt_metric_delta(val_m['fix_rrec'],
                                         val_baseline['t2n_rrec'])

            print(f"Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  "
                  f"(full={avg_full:.4f}  fix={avg_fix:.4f})")
            print(f"  Train full: rIoU={t_riou_d}  rRec={t_rrec_d}")
            print(f"  Train fix:  rIoU={tf_riou_d}  rRec={tf_rrec_d}")
            print(f"  Val   full: rIoU={v_riou_d}  rRec={v_rrec_d}")
            print(f"  Val   fix:  rIoU={vf_riou_d}  rRec={vf_rrec_d}")
        elif (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  "
                  f"(full={avg_full:.4f}  fix={avg_fix:.4f})")

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()

    print(f"\n── Final Summary ──")
    print(f"Loss reduction: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f} "
          f"({(1 - history['loss'][-1]/history['loss'][0])*100:.1f}% decrease)")
    print(f"Best val fix_riou: {best_val_riou:.4f} at epoch {best_epoch}")

    # Final full val evaluation
    print(f"\nEvaluating full validation set ({len(val_ids)} tiles)...")
    val_final = evaluate_tiles(model, val_ids, device,
                               zero_channels=zero_channels,
                               tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                               conf_dir=conf_dir, gt_dir=gt_dir)
    print(f"Val (all {len(val_ids)} tiles):")
    print(f"  T2N baseline:  rIoU={val_final['t2n_riou']:.4f}  "
          f"rRec={val_final['t2n_rrec']:.4f}")
    print(f"  Full head:     rIoU="
          f"{fmt_metric_delta(val_final['full_riou'], val_final['t2n_riou'])}  "
          f"rRec="
          f"{fmt_metric_delta(val_final['full_rrec'], val_final['t2n_rrec'])}")
    print(f"  Fix head:      rIoU="
          f"{fmt_metric_delta(val_final['fix_riou'], val_final['t2n_riou'])}  "
          f"rRec="
          f"{fmt_metric_delta(val_final['fix_rrec'], val_final['t2n_rrec'])}")

    return history, train_baseline, val_baseline, val_final


# ──────────────────────────────────────────────
# Export predicted masks to disk
# ──────────────────────────────────────────────

def export_predictions(model, tile_ids, device,
                       export_dir="/Users/stefancobeli/Desktop/Research/"
                       "GNN_Sidewalks/tile2net-main-private/tile2net_data_files/"
                       "boston_data_sources/dorchester_exports",
                       tiles_dir=None, t2n_dir=None, conf_dir=None,
                       gt_dir=None):
    """Export model predictions as mask PNGs alongside original data.

    Creates two folders under export_dir:
        masks_full_prediction/
            Binary masks from the full head (threshold 0.5).
            256×256 grayscale PNG, 0/255 values.

        masks_fix_prediction/
            Binary masks from the fix head OR'd with T2N input.
            256×256 grayscale PNG, 0/255 values.

        masks_fix_prediction_overlay/
            RGB overlay: T2N pixels in white, new fix pixels in
            pastel green (#C1E1C1), background in black.
            Makes it easy to see what the model added.

    File naming: {tile_id}.png (standardized, no suffix).

    Parameters
    ----------
    model      : ResidualFixNet — already on device, will be set to eval
    tile_ids   : list of str — tiles to export
    device     : torch.device
    export_dir : str — parent directory for output folders
    tiles_dir  : str or None — override default tiles folder
    t2n_dir    : str or None — override default T2N masks folder
    conf_dir   : str or None — override default confidence masks folder
    gt_dir     : str or None — override default ground truth masks folder
    """
    import os
    from PIL import Image

    model.eval()
    ds = SidewalkFixDataset(tile_ids, tiles_dir=tiles_dir, t2n_dir=t2n_dir,
                            conf_dir=conf_dir, gt_dir=gt_dir)

    dir_full = os.path.join(export_dir, "masks_full_prediction")
    dir_fix = os.path.join(export_dir, "masks_fix_prediction")
    dir_overlay = os.path.join(export_dir, "masks_fix_prediction_overlay")

    os.makedirs(dir_full, exist_ok=True)
    os.makedirs(dir_fix, exist_ok=True)
    os.makedirs(dir_overlay, exist_ok=True)

    for i in range(len(ds)):
        inp, _, _, tid = ds[i]

        with torch.no_grad():
            outputs = model(inp.unsqueeze(0).to(device))
        has_remove = hasattr(model, 'enable_remove') and model.enable_remove
        if has_remove:
            pred_full, pred_fix, pred_remove = outputs
            pred_remove_np = pred_remove[0, 0].cpu().numpy()
        else:
            pred_full, pred_fix = outputs
            pred_remove_np = None
        pred_full_np = pred_full[0, 0].cpu().numpy()
        pred_fix_np = pred_fix[0, 0].cpu().numpy()

        inp_np = inp.numpy()
        t2n_binary = inp_np[3] > 0.5

        # Full head: threshold at 0.5
        full_mask = (pred_full_np > 0.5).astype(np.uint8) * 255

        # Fix head: remove first, then add
        fix_new = pred_fix_np > 0.5
        if pred_remove_np is not None:
            fix_combined = ((t2n_binary & ~(pred_remove_np > 0.5)) | fix_new).astype(np.uint8) * 255
        else:
            fix_combined = (t2n_binary | fix_new).astype(np.uint8) * 255

        # Overlay: T2N = white, new fixes = pastel green, removed = red, bg = black
        overlay = np.zeros((256, 256, 3), dtype=np.uint8)
        # T2N pixels → white
        overlay[t2n_binary] = [255, 255, 255]
        # Removed pixels (in T2N but removed) → pastel red
        if pred_remove_np is not None:
            removed = t2n_binary & (pred_remove_np > 0.5)
            overlay[removed] = [225, 150, 150]
        # New fix pixels (not in T2N) → pastel green #C1E1C1
        new_only = fix_new & ~t2n_binary
        overlay[new_only] = [193, 225, 193]

        fname = f"{tid}.png"
        Image.fromarray(full_mask, mode="L").save(
            os.path.join(dir_full, fname))
        Image.fromarray(fix_combined, mode="L").save(
            os.path.join(dir_fix, fname))
        Image.fromarray(overlay, mode="RGB").save(
            os.path.join(dir_overlay, fname))

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Exported {i + 1}/{len(ds)} tiles...")

    print(f"\nExported {len(ds)} tiles to:")
    print(f"  Full head masks:    {dir_full}")
    print(f"  Fix head masks:     {dir_fix}")
    print(f"  Fix head overlays:  {dir_overlay}")


# ──────────────────────────────────────────────
# Multi-run comparison
# ──────────────────────────────────────────────

def plot_multi_run_bars(run_results, head="fix", run_names=None, plot_title="Val performance by bucket and metric"):
    """Grouped bar chart: T2N baseline vs trained models, per bucket, per metric.

    X-axis groups = metrics (rRec, Rec, rIoU, IoU).
    Within each group: one bar cluster per bucket (colored by hardness).
    Overlaid bars per cluster: T2N solid fill, each model run with a
    distinct overlay style (transparent, hatched, etc.).

    Parameters
    ----------
    run_results : list of dict, each with keys:
        'n_tiles'     : int or str — label for this run
        'val_metrics' : dict — bucket_metrics dict keyed by bucket label,
                        each value is the flat dict returned by evaluate_tiles()
                        e.g. {"[0, 0.25)": {"t2n_rrec": 0.3, "fix_rrec": 0.4, ...}, ...}
    head      : str — "fix" or "full" — which head's metrics to display
    run_names : list of str, optional — display names for each run
                (default: "<n_tiles> tiles")
    """
    from matplotlib.patches import Patch

    metric_names = ["rRec", "Rec", "rIoU", "IoU"]
    t2n_keys     = ["t2n_rrec", "t2n_rec", "t2n_riou", "t2n_iou"]
    fix_keys     = [f"{head}_rrec", f"{head}_rec", f"{head}_riou", f"{head}_iou"]

    b_labels  = list(run_results[0]["val_metrics"].keys())
    n_buckets = len(b_labels)
    n_metrics = len(metric_names)
    n_runs    = len(run_results)

    if run_names is None:
        run_names = [
            f"{r['n_tiles']} tiles" if isinstance(r['n_tiles'], int) else str(r['n_tiles'])
            for r in run_results
        ]

    bucket_colors = ["#F4A7A0", "#F7C9A5", "#A7C7E7", "#A8D8B9"]
    bar_width = 0.16

    # Overlay style per model run: first = transparent solid, second = hatched dashed, etc.
    _run_styles = [
        dict(facecolor_alpha=0.45, edgewidth=1.5, linestyle="-",  hatch=None),
        dict(facecolor_alpha=None,  edgewidth=1.8, linestyle="--", hatch="///"),
        dict(facecolor_alpha=0.60,  edgewidth=1.5, linestyle="-.", hatch=None),
        dict(facecolor_alpha=None,  edgewidth=1.8, linestyle=":",  hatch="..."),
    ]

    fig, ax = plt.subplots(figsize=(15, 7))

    for j, (mname, tk, fk) in enumerate(zip(metric_names, t2n_keys, fix_keys)):
        group_center = j
        for b_idx, (blabel, color) in enumerate(zip(b_labels, bucket_colors)):
            x = group_center + (b_idx - (n_buckets - 1) / 2) * bar_width

            # T2N baseline — use first run (T2N output is model-independent)
            t2n_val  = run_results[0]["val_metrics"][blabel][tk]
            fix_vals = [r["val_metrics"][blabel][fk] for r in run_results]

            # Bar 1: T2N baseline — solid fill
            ax.bar(x, t2n_val, width=bar_width * 0.9, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3)

            # Model run bars — overlaid with distinct styles
            for r_idx, (fix_val, sty) in enumerate(zip(fix_vals, _run_styles)):
                if sty["hatch"] is None:
                    ax.bar(x, fix_val, width=bar_width * 0.9, color=color,
                           alpha=sty["facecolor_alpha"], edgecolor=color,
                           linewidth=sty["edgewidth"], linestyle=sty["linestyle"],
                           zorder=4 + r_idx)
                else:
                    ax.bar(x, fix_val, width=bar_width * 0.9,
                           facecolor="none", edgecolor=color,
                           linewidth=sty["edgewidth"], linestyle=sty["linestyle"],
                           hatch=sty["hatch"], zorder=4 + r_idx)

            # Annotations: stagger horizontally across runs
            top = max(t2n_val, *fix_vals)
            for r_idx, fix_val in enumerate(fix_vals):
                label_char = chr(ord('A') + r_idx)
                delta = fix_val - t2n_val
                sign  = "+" if delta >= 0 else ""
                x_off = (r_idx - (n_runs - 1) / 2) * bar_width * 0.28
                y_off = 0.015 + r_idx * 0.04
                ax.text(x + x_off, top + y_off,
                        f"{label_char}:{fix_val:.3f}({sign}{delta:.3f})",
                        ha="center", va="bottom", fontsize=5.5, fontweight="bold",
                        color="#555555" if r_idx % 2 == 0 else "#222222")

            # T2N value below bars
            ax.text(x, min(t2n_val, *fix_vals) - 0.05, f"{t2n_val:.3f}",
                    ha="center", va="bottom", fontsize=6, color="#333333")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.09)
    ax.grid(True, alpha=0.2, axis="y")

    ax.set_title(f"{plot_title}",
                 fontweight="bold")
    # ax.set_title(f"Val performance by bucket and metric ({head} head)",
                #  fontweight="bold")

    # Legend
    legend_elems = []
    for blabel, color in zip(b_labels, bucket_colors):
        legend_elems.append(Patch(facecolor=color, label=f"Bucket {blabel}"))
    legend_elems.append(Patch(facecolor="gray", edgecolor="white",
                              linewidth=0.8, label="T2N baseline (solid)"))
    for r_idx, (name, sty) in enumerate(zip(run_names, _run_styles)):
        label_char = chr(ord('A') + r_idx)
        if sty["hatch"] is None:
            legend_elems.append(Patch(facecolor="gray",
                                      alpha=sty["facecolor_alpha"],
                                      label=f"{label_char}: {name}"))
        else:
            legend_elems.append(Patch(facecolor="none", edgecolor="gray",
                                      hatch=sty["hatch"],
                                      label=f"{label_char}: {name}"))

    ax.legend(handles=legend_elems, fontsize=8, loc="upper right",
              ncol=2 if n_runs > 1 else 1)
    plt.tight_layout()
    plt.show()

import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from PIL import Image
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
import math
from typing import Tuple
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ── Tile coordinate helpers (same as generate_groundtruth_masks_from_polygons) ──

def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon (top-left corner)."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_tile_bounds(xtile: int, ytile: int, zoom: int) -> Tuple[float, float, float, float]:
    """Get geographic bounding box (west, south, east, north) in EPSG:4326."""
    lat_top, lon_left = num2deg(xtile, ytile, zoom)
    lat_bottom, lon_right = num2deg(xtile + 1, ytile + 1, zoom)
    return lon_left, lat_bottom, lon_right, lat_top


def rasterize_polygons_to_mask(gdf, xtile, ytile, zoom=19, resolution=256):
    """Rasterize polygons to a binary mask for a specific tile."""
    west, south, east, north = get_tile_bounds(xtile, ytile, zoom)
    gdf_clipped = gdf.cx[west:east, south:north]

    if len(gdf_clipped) == 0:
        return np.zeros((resolution, resolution), dtype=np.uint8)

    transform = from_bounds(west, south, east, north, resolution, resolution)
    shapes = [(geom, 1) for geom in gdf_clipped.geometry
              if geom is not None and geom.is_valid]

    if len(shapes) == 0:
        return np.zeros((resolution, resolution), dtype=np.uint8)

    mask = features.rasterize(
        shapes=shapes,
        out_shape=(resolution, resolution),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return (mask > 0).astype(np.uint8) * 255


print("\u2713 Helper functions defined")


import torch
import sys
sys.path.insert(0, ".")
from tile2net_training_utils import ResidualFixNet

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Load trained model ──
# Option A: Load from the training notebook (run training notebook first,
#           then save model with torch.save, then load here).
# Option B: Train fresh here.
# For now, train a quick model here using the same setup as the training notebook.

# from tile2net_training_utils import (
#     SidewalkFixDataset, train_model, stratified_split,
#     get_non_empty_tile_ids, export_predictions,
# )
from tile2net_gt_utils import ALL_TILE_IDS, load_tile_data
import geopandas as gpd

# ── Tile coordinate helpers (same as generate_groundtruth_masks_from_polygons) ──

from shapely.geometry import LineString, MultiLineString, box
from shapely import clip_by_rect
from tile2net_training_utils import compute_refined_metrics, fmt_metric_delta
from tile2net_gt_utils import compute_iou, compute_recall


def _geo_to_px(coords, bounds, resolution=256):
    """Convert geographic coordinates to pixel coordinates."""
    west, south, east, north = bounds
    px_coords = []
    for x, y in coords:
        px_x = (x - west) / (east - west) * resolution
        px_y = (north - y) / (north - south) * resolution
        px_coords.append((px_x, px_y))
    return px_coords


def _plot_line_overlay(ax, geom, bounds, color="red", linewidth=1.5,
                       resolution=256):
    """Plot LineString / MultiLineString on a pixel-coordinate axis."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        if len(geom.coords) < 2:
            return
        px = _geo_to_px(geom.coords, bounds, resolution)
        xs, ys = zip(*px)
        ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle="round")
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            _plot_line_overlay(ax, part, bounds, color, linewidth, resolution)
    elif hasattr(geom, "geoms"):  # GeometryCollection
        for part in geom.geoms:
            _plot_line_overlay(ax, part, bounds, color, linewidth, resolution)


def _load_network(path, label="network"):
    """Load a network shapefile or GeoJSON, reproject to 4326, filter to sidewalk."""
    print(f"Loading {label}: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    n_all = len(gdf)
    # Filter to sidewalk only
    if "f_type" in gdf.columns:
        gdf = gdf[gdf["f_type"] == "sidewalk"].copy()
    print(f"  \u2192 {len(gdf):,}/{n_all:,} sidewalk features, geom types: "
          f"{gdf.geom_type.value_counts().to_dict()}")
    return gdf


def _clip_gdf_to_tile(gdf, bounds):
    """Spatially index + clip geometries to tile bounds."""
    west, south, east, north = bounds
    # Fast bbox pre-filter
    subset = gdf.cx[west:east, south:north]
    if len(subset) == 0:
        return subset
    # Precise clip to tile rectangle
    clipped_geoms = []
    keep_idx = []
    for idx, row in subset.iterrows():
        clipped = clip_by_rect(row.geometry, west, south, east, north)
        if clipped is not None and not clipped.is_empty:
            clipped_geoms.append(clipped)
            keep_idx.append(idx)
    if not clipped_geoms:
        return subset.iloc[:0]  # empty GeoDataFrame with same schema
    result = subset.loc[keep_idx].copy()
    result.geometry = clipped_geoms
    return result


def display_polygon_network(tile_ids,
                            tiles_dir,
                            conf_dir,
                            orig_mask_dir,
                            orig_network_shp,
                            mod_mask_dir,
                            mod_network_shp,
                            gt_mask_dir=None,
                            gt_network_path=None,
                            show_satellite=False,
                            network_color_orig="cyan",
                            network_color_mod="lime",
                            network_color_gt="orange",
                            network_linewidth=1.5,
                            title="Polygon Mask + Network Overlay",
                            subtitles=None,
                            figscale=3.5,
                            zoom=19):
    """Display tiles comparing original vs modified vs GT polygon masks with network overlays.

    Columns:
        1. Satellite (RGB)
        2. Confidence heatmap
        3. Original polygon mask + original network lines  (metrics vs GT)
        4. Modified polygon mask + modified network lines  (metrics + delta vs original)
        5. GT polygon mask + GT network lines  (only if gt_mask_dir and gt_network_path provided)

    Parameters
    ----------
    tile_ids : list of str
    tiles_dir : str or Path \u2014 satellite tiles folder
    conf_dir : str or Path \u2014 confidence masks folder
    orig_mask_dir : str or Path \u2014 original polygon mask folder
    orig_network_shp : str or Path \u2014 original line-network shapefile (filtered to sidewalk)
    mod_mask_dir : str or Path \u2014 modified polygon mask folder
    mod_network_shp : str or Path \u2014 modified line-network shapefile (filtered to sidewalk)
    gt_mask_dir : str or Path or None \u2014 GT polygon mask folder (.png files)
    gt_network_path : str or Path or None \u2014 GT line-network file (.geojson, filtered to sidewalk)
    show_satellite : bool \u2014 blend satellite behind polygon mask (default False)
    network_color_orig : str \u2014 line color for original network (default "cyan")
    network_color_mod : str \u2014 line color for modified network (default "lime")
    network_color_gt : str \u2014 line color for GT network (default "orange")
    network_linewidth : float \u2014 line width for network lines
    title : str \u2014 figure title
    subtitles : list of str or None \u2014 per-row subtitles
    figscale : float \u2014 size multiplier per subplot
    zoom : int \u2014 tile zoom level (default 19)
    """
    tiles_dir = Path(tiles_dir)
    conf_dir = Path(conf_dir)
    orig_mask_dir = Path(orig_mask_dir)
    mod_mask_dir = Path(mod_mask_dir)

    show_gt = gt_mask_dir is not None and gt_network_path is not None
    if show_gt:
        gt_mask_dir = Path(gt_mask_dir)

    # Load network files (once, then clip per tile)
    gdf_orig_net = _load_network(orig_network_shp, "original network")
    gdf_mod_net = _load_network(mod_network_shp, "modified network")
    gdf_gt_net = _load_network(gt_network_path, "GT network") if show_gt else None

    n_rows = len(tile_ids)
    n_cols = 5 if show_gt else 4
    col_titles = ["Satellite", "Confidence",
                  "Original (mask + network)", "Modified (mask + network)"]
    if show_gt:
        col_titles.append("GT (mask + network)")

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * figscale, figscale * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, tid in enumerate(tile_ids):
        xtile, ytile = map(int, tid.split("_"))
        bounds = get_tile_bounds(xtile, ytile, zoom)
        west, south, east, north = bounds

        # \u2500\u2500 Load data \u2500\u2500
        sat = np.array(Image.open(tiles_dir / f"{tid}.jpg").convert("RGB"))

        conf_path = conf_dir / f"{tid}.png"
        conf_ch = (np.array(Image.open(conf_path).convert("L")).astype(np.float32) / 255.0
                   if conf_path.exists()
                   else np.zeros((256, 256), dtype=np.float32))

        orig_mask_path = orig_mask_dir / f"{tid}.png"
        orig_mask = (np.array(Image.open(orig_mask_path).convert("L"))
                     if orig_mask_path.exists()
                     else np.zeros((256, 256), dtype=np.uint8))

        mod_mask_path = mod_mask_dir / f"{tid}.png"
        mod_mask = (np.array(Image.open(mod_mask_path).convert("L"))
                    if mod_mask_path.exists()
                    else np.zeros((256, 256), dtype=np.uint8))

        # GT mask (needed for metrics even if not displaying GT column)
        gt_mask_arr = np.zeros((256, 256), dtype=np.uint8)
        if show_gt:
            gt_mask_path = gt_mask_dir / f"{tid}.png"
            if gt_mask_path.exists():
                gt_mask_arr = np.array(Image.open(gt_mask_path).convert("L"))
        gt_bin = gt_mask_arr > 127

        # Binary masks for metrics
        orig_bin = orig_mask > 127
        mod_bin = mod_mask > 127

        # \u2500\u2500 Compute metrics: Original vs GT \u2500\u2500
        orig_riou, orig_rrec, _, _ = compute_refined_metrics(orig_bin, gt_bin)
        orig_iou = compute_iou(orig_bin, gt_bin)
        orig_rec = compute_recall(orig_bin, gt_bin)

        # \u2500\u2500 Compute metrics: Modified vs GT \u2500\u2500
        mod_riou, mod_rrec, _, _ = compute_refined_metrics(mod_bin, gt_bin)
        mod_iou = compute_iou(mod_bin, gt_bin)
        mod_rec = compute_recall(mod_bin, gt_bin)

        # Clip network features to tile bounds
        orig_clipped = _clip_gdf_to_tile(gdf_orig_net, bounds)
        mod_clipped = _clip_gdf_to_tile(gdf_mod_net, bounds)

        # \u2500\u2500 Row label \u2500\u2500
        ylabel = tid if subtitles is None else f"{tid}\n{subtitles[i]}"
        axes[i, 0].set_ylabel(ylabel, fontsize=8, rotation=90,
                               labelpad=10, va="center")

        # \u2500\u2500 Col 0: Satellite \u2500\u2500
        axes[i, 0].imshow(sat)

        # \u2500\u2500 Col 1: Confidence \u2500\u2500
        axes[i, 1].imshow(conf_ch, cmap="hot", vmin=0, vmax=1)

        # \u2500\u2500 Helper to render mask + network on an axis \u2500\u2500
        def _render_mask_network(ax, mask, clipped_net, net_color):
            if show_satellite:
                bg = sat.copy().astype(np.float32) * 0.4
                mask_rgb = np.stack([mask] * 3, axis=-1).astype(np.float32) * 0.6
                blended = np.clip(bg + mask_rgb, 0, 255).astype(np.uint8)
                ax.imshow(blended)
            else:
                ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
            for _, row in clipped_net.iterrows():
                _plot_line_overlay(ax, row.geometry, bounds,
                                   color=net_color,
                                   linewidth=network_linewidth)
            # Ensure axes stay within tile pixel bounds
            ax.set_xlim(0, 256)
            ax.set_ylim(256, 0)

        # \u2500\u2500 Col 2: Original mask + network + metrics \u2500\u2500
        _render_mask_network(axes[i, 2], orig_mask, orig_clipped,
                             network_color_orig)
        axes[i, 2].set_xlabel(
            f"rIoU={orig_riou:.2f}  IoU={orig_iou:.2f}\n"
            f"rRec={orig_rrec:.2f}  Rec={orig_rec:.2f}\n"
            f"{len(orig_clipped)} segment(s)", fontsize=7)

        # \u2500\u2500 Col 3: Modified mask + network + metrics with delta \u2500\u2500
        _render_mask_network(axes[i, 3], mod_mask, mod_clipped,
                             network_color_mod)
        axes[i, 3].set_xlabel(
            f"rIoU={fmt_metric_delta(mod_riou, orig_riou)}  "
            f"IoU={fmt_metric_delta(mod_iou, orig_iou)}\n"
            f"rRec={fmt_metric_delta(mod_rrec, orig_rrec)}  "
            f"Rec={fmt_metric_delta(mod_rec, orig_rec)}\n"
            f"{len(mod_clipped)} segment(s)", fontsize=7)

        # \u2500\u2500 Col 4: GT mask + network (optional) \u2500\u2500
        if show_gt:
            gt_clipped = _clip_gdf_to_tile(gdf_gt_net, bounds)
            _render_mask_network(axes[i, 4], gt_mask_arr, gt_clipped,
                                 network_color_gt)
            axes[i, 4].set_xlabel(
                f"{len(gt_clipped)} segment(s)", fontsize=7)

    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

