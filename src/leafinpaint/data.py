"""Synthetic leaves and damage masks.

Real leaf damage (insect feeding, shot-hole disease, tears) removes tissue,
so the background shows through the hole. To measure a restoration we need
the leaf as it was before the damage, which never exists for real photos.
Here the intact leaf is generated first and the damage is applied after, so
the ground truth is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import draw


def _noise(shape, sigma, rng):
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / (np.abs(n).max() + 1e-9)


@dataclass
class Leaf:
    rgb: np.ndarray  # intact leaf on its background, H x W x 3 float in [0, 1]
    mask: np.ndarray  # bool, leaf pixels
    background: np.ndarray  # the background alone, used to fill holes


def make_leaf(size: int = 192, seed: int | None = None) -> Leaf:
    """A textured leaf with a midrib and secondary veins on a light, slightly uneven background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    angle = rng.uniform(-0.5, 0.5)
    c = size / 2
    x = (xx - c) * np.cos(angle) + (yy - c) * np.sin(angle)
    y = -(xx - c) * np.sin(angle) + (yy - c) * np.cos(angle)
    length = size * rng.uniform(0.36, 0.43)
    width = length * rng.uniform(0.4, 0.62)
    t = np.clip((y + length) / (2 * length), 0, 1)
    profile = width * np.sin(np.pi * t ** rng.uniform(0.75, 0.95)) ** 0.8
    serration = 1 + 0.03 * np.sin(t * rng.integers(14, 26) * np.pi)  # small teeth on the margin
    mask = (np.abs(x) <= profile * serration) & (np.abs(y) <= length)

    bg = np.array([0.88, 0.87, 0.83]) * (1 + 0.08 * ((xx / size) - 0.5))[..., None]
    bg = bg + 0.015 * _noise((size, size), 2, rng)[..., None]

    base = np.array([0.22, 0.47, 0.18]) * rng.uniform(0.9, 1.1)
    tex = 0.05 * _noise((size, size), 2.5, rng)[..., None] + 0.03 * _noise((size, size), 0.8, rng)[..., None]
    leaf = base + tex
    # midrib along x == 0 and secondary veins at an angle
    midrib = np.exp(-(x**2) / 2.5)
    phase = (y + np.abs(x) * 0.9) / (length / rng.uniform(5, 8))
    secondary = np.exp(-((phase - np.round(phase)) ** 2) / 0.004) * (np.abs(x) > 2)
    vein = np.clip(midrib + 0.6 * secondary, 0, 1)[..., None]
    leaf = leaf * (1 - vein) + np.array([0.45, 0.66, 0.32]) * vein
    alpha = ndi.gaussian_filter(mask.astype(float), 0.6)[..., None]
    rgb = np.clip(bg * (1 - alpha) + leaf * alpha, 0, 1)
    return Leaf(rgb, mask, np.clip(bg, 0, 1))


def bite_mask(leaf_mask: np.ndarray, n: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    """Insect feeding from the margin: round bites centred on the leaf edge."""
    rng = rng or np.random.default_rng()
    size = leaf_mask.shape[0]
    edge = leaf_mask & ~ndi.binary_erosion(leaf_mask)
    ys, xs = np.nonzero(edge)
    out = np.zeros_like(leaf_mask)
    for _ in range(n):
        k = rng.integers(len(ys))
        r = rng.uniform(0.04, 0.08) * size
        rr, cc = draw.disk((ys[k], xs[k]), r, shape=out.shape)
        out[rr, cc] = True
    return out & ndi.binary_dilation(leaf_mask, iterations=2)


def hole_mask(leaf_mask: np.ndarray, n: int = 5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Shot-hole damage: small holes inside the blade."""
    rng = rng or np.random.default_rng()
    size = leaf_mask.shape[0]
    inner = ndi.binary_erosion(leaf_mask, iterations=int(0.06 * size))
    ys, xs = np.nonzero(inner if inner.any() else leaf_mask)
    out = np.zeros_like(leaf_mask)
    for _ in range(n):
        k = rng.integers(len(ys))
        r = rng.uniform(0.015, 0.04) * size
        rr, cc = draw.ellipse(ys[k], xs[k], r, r * rng.uniform(0.6, 1.4), shape=out.shape, rotation=rng.uniform(0, np.pi))
        out[rr, cc] = True
    return out


def tear_mask(leaf_mask: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """A tear: a thick irregular stroke from the margin into the blade (like LaMa's free-form masks)."""
    rng = rng or np.random.default_rng()
    size = leaf_mask.shape[0]
    edge = leaf_mask & ~ndi.binary_erosion(leaf_mask)
    ys, xs = np.nonzero(edge)
    k = rng.integers(len(ys))
    y, x = float(ys[k]), float(xs[k])
    cy, cx = np.argwhere(leaf_mask).mean(axis=0)
    ang = np.arctan2(cy - y, cx - x)
    out = np.zeros_like(leaf_mask)
    width = rng.uniform(0.02, 0.035) * size
    for _ in range(int(0.3 * size)):
        rr, cc = draw.disk((y, x), width, shape=out.shape)
        out[rr, cc] = True
        ang += rng.normal(0, 0.15)
        y, x = y + np.sin(ang), x + np.cos(ang)
    return out & ndi.binary_dilation(leaf_mask, iterations=2)


DAMAGE = {"bites": bite_mask, "holes": hole_mask, "tear": tear_mask}


@dataclass
class DamagedLeaf:
    intact: Leaf
    damaged: np.ndarray  # rgb with background showing through the damage
    damage: np.ndarray  # bool, the region to restore


def damage_leaf(leaf: Leaf, kind: str, seed: int | None = None) -> DamagedLeaf:
    rng = np.random.default_rng(seed)
    region = DAMAGE[kind](leaf.mask, rng=rng)
    damaged = np.where(region[..., None], leaf.background, leaf.rgb)
    return DamagedLeaf(leaf, damaged, region)
