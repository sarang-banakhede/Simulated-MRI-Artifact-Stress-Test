from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

METRIC_KEYS = ["dsc", "iou", "precision", "recall", "hd95", "nsd", "gt_ml", "pred_ml", "empty_pred"]


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing, tol: float) -> tuple[float, float]:
    """HD95 (mm) and normalized surface Dice at tolerance `tol` (mm) between two binary volumes."""
    idx = np.argwhere(pred | gt)
    lo, hi = np.maximum(idx.min(0) - 2, 0), np.minimum(idx.max(0) + 3, pred.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    p, g = pred[sl], gt[sl]
    st = generate_binary_structure(3, 1)
    sp = p & ~binary_erosion(p, structure=st)
    sg = g & ~binary_erosion(g, structure=st)
    d_pg = distance_transform_edt(~sg, sampling=spacing)[sp]
    d_gp = distance_transform_edt(~sp, sampling=spacing)[sg]
    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    allv = np.concatenate([d_pg, d_gp])
    return hd95, float((allv <= tol).sum() / allv.size)


def case_metrics(pred: np.ndarray, gt: np.ndarray, spacing, diag: float, nsd_tol: float) -> dict:
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    gt_n, pr_n = tp + fn, tp + fp
    vox_ml = float(np.prod(spacing)) / 1000.0
    if gt_n == 0 and pr_n == 0:
        dsc = iou = precision = recall = nsd = 1.0
        hd95 = 0.0
    else:
        dsc = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        precision = tp / pr_n if pr_n > 0 else 0.0
        recall = tp / gt_n if gt_n > 0 else 0.0
        if gt_n > 0 and pr_n > 0:
            hd95, nsd = surface_metrics(pred, gt, spacing, nsd_tol)
        else:
            hd95, nsd = diag, 0.0
    return {
        "dsc": dsc,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "hd95": hd95,
        "nsd": nsd,
        "gt_ml": gt_n * vox_ml,
        "pred_ml": pr_n * vox_ml,
        "empty_pred": int(pr_n == 0),
    }
