"""How good is a restoration? Pixel metrics and a leaf-shape metric.

PSNR and SSIM reward images that *look* close to the original. For leaves
that is not enough: a method can produce plausible green texture in a
place where the leaf never was, or leave a bite in the margin, and the
pixel scores barely change because the hole is a small part of the image.
``shape_iou`` checks the thing a biologist would measure afterwards: is the
restored leaf outline where the real one was?
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(restored: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR in dB; with ``mask`` only the restored pixels count."""
    if mask is None:
        return float(peak_signal_noise_ratio(truth, restored, data_range=1.0))
    mse = float(((restored[mask] - truth[mask]) ** 2).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(restored: np.ndarray, truth: np.ndarray) -> float:
    return float(structural_similarity(truth, restored, channel_axis=-1, data_range=1.0))


def leaf_mask(rgb: np.ndarray, fill_holes: bool = False) -> np.ndarray:
    """Excess-green + Otsu leaf segmentation (as in plant-cell-image-analysis).

    Holes are *not* filled by default: an unrestored hole is missing tissue.
    """
    total = rgb.sum(axis=2) + 1e-6
    exg = (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) / total
    mask = exg > threshold_otsu(exg)
    return ndi.binary_fill_holes(mask) if fill_holes else mask


def shape_iou(restored: np.ndarray, truth_mask: np.ndarray, damage: np.ndarray, margin: int = 3) -> float:
    """IoU between the restored and the true leaf tissue, inside the damaged area (plus a small margin).

    Measured locally because the undamaged part of the leaf is trivially right.
    Both a bite left in the margin and green texture painted onto the
    background lower it.
    """
    region = ndi.binary_dilation(damage, iterations=margin)
    pred = leaf_mask(restored) & region
    true = truth_mask & region
    union = (pred | true).sum()
    return float((pred & true).sum() / union) if union else 1.0


def leaf_area_error(restored: np.ndarray, truth_mask: np.ndarray) -> float:
    """Relative error in total leaf area, the number a phenotyping pipeline would report."""
    return float((leaf_mask(restored).sum() - truth_mask.sum()) / truth_mask.sum())
