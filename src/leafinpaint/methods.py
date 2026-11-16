"""Inpainting methods, from the simplest to exemplar-based.

All take an RGB float image in [0, 1] and a boolean mask (True = fill) and
return the restored image. The known pixels are never changed.

* ``harmonic``    - solves Laplace's equation in the hole (smoothest possible fill);
                    written from scratch as one sparse linear system.
* ``biharmonic``  - scikit-image's biharmonic fill, smoother continuation of gradients.
* ``telea``       - OpenCV fast-marching method (Telea 2004).
* ``navier_stokes`` - OpenCV fluid-dynamics method (Bertalmio et al. 2001).
* ``exemplar``    - Criminisi, Perez & Toyama (2004), written from scratch: copies
                    whole patches from the known region, filling along strong edges first,
                    so texture and structure (veins, leaf margin) are continued.

Diffusion methods blur texture; exemplar-based ones keep texture but can paste
the wrong thing. Deep models (LaMa, Stable Diffusion inpainting, BrushNet) fit
in through ``register``; see ``deep.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

Inpainter = Callable[[np.ndarray, np.ndarray], np.ndarray]
METHODS: dict[str, Inpainter] = {}


def register(name: str):
    def wrap(fn: Inpainter) -> Inpainter:
        METHODS[name] = fn
        return fn
    return wrap


def _check(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(img, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if img.shape[:2] != mask.shape:
        raise ValueError(f"image {img.shape[:2]} and mask {mask.shape} differ in size")
    if mask.all():
        raise ValueError("nothing is known: the mask covers the whole image")
    return img, mask


@register("harmonic")
def harmonic(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Each unknown pixel equals the mean of its 4 neighbours (discrete Laplace equation).

    Known neighbours become the right-hand side, so the system only has as many
    unknowns as there are masked pixels.
    """
    img, mask = _check(img, mask)
    h, w = mask.shape
    idx = -np.ones(mask.shape, np.int64)
    ys, xs = np.nonzero(mask)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, vals = [], [], []
    rhs = np.zeros((len(ys), img.shape[2]))
    deg = np.zeros(len(ys))
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        deg += inside
        k = np.nonzero(inside)[0]
        nb_unknown = mask[ny[k], nx[k]]
        u = k[nb_unknown]
        rows.append(u); cols.append(idx[ny[u], nx[u]]); vals.append(-np.ones(len(u)))
        kn = k[~nb_unknown]
        rhs[kn] += img[ny[kn], nx[kn]]
    n = len(ys)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(deg)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    out = img.copy()
    solution = spsolve(A, rhs)
    out[ys, xs] = solution.reshape(n, -1)
    return np.clip(out, 0, 1)


@register("biharmonic")
def biharmonic(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from skimage.restoration import inpaint_biharmonic

    img, mask = _check(img, mask)
    return np.clip(inpaint_biharmonic(img, mask, channel_axis=-1), 0, 1)


def _opencv(img, mask, flag):
    import cv2

    img, mask = _check(img, mask)
    u8 = (img * 255).round().astype(np.uint8)
    out = cv2.inpaint(u8, mask.astype(np.uint8), 5, flag)
    return out.astype(float) / 255


@register("telea")
def telea(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import cv2

    return _opencv(img, mask, cv2.INPAINT_TELEA)


@register("navier_stokes")
def navier_stokes(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import cv2

    return _opencv(img, mask, cv2.INPAINT_NS)


@register("exemplar")
def exemplar(img: np.ndarray, mask: np.ndarray, patch: int = 9, search: int = 40) -> np.ndarray:
    """Criminisi et al. (2004) exemplar-based inpainting.

    Repeat until the hole is filled:
      1. On the fill front, priority P(p) = C(p) * D(p):
         C = average confidence of the patch (how much of it is known or was filled early),
         D = |isophote . normal| / alpha (strong edges that run *into* the hole go first).
      2. For the highest-priority patch, find the fully known patch within ``search``
         pixels with the smallest sum of squared differences over the known pixels.
      3. Copy its pixels into the unknown part and propagate the confidence.
    Restricting the search to a window keeps it fast and favours nearby texture.
    """
    img, mask = _check(img, mask)
    out = img.copy()
    unknown = mask.copy()
    h, w = unknown.shape
    r = patch // 2
    confidence = (~unknown).astype(float)
    gray = out.mean(axis=2)

    def known_source_centres() -> np.ndarray:
        # centres whose whole patch lies in the originally known region
        bad = ndi.binary_dilation(mask, structure=np.ones((patch, patch)))
        valid = ~bad
        valid[:r, :] = valid[-r:, :] = valid[:, :r] = valid[:, -r:] = False
        return valid

    source_ok = known_source_centres()
    if not source_ok.any():
        raise ValueError("no fully known patch to copy from; use a smaller patch size")

    alpha = 1.0
    while unknown.any():
        front = unknown & ndi.binary_dilation(~unknown)
        fy, fx = np.nonzero(front)
        # confidence term
        conf_sum = ndi.uniform_filter(confidence, size=patch, mode="constant") * patch * patch
        C = conf_sum[fy, fx] / (patch * patch)
        # data term: isophote (rotated gradient of the known image) dotted with the front normal
        g = np.where(unknown, np.nan, gray)
        gy, gx = np.gradient(np.nan_to_num(g, nan=np.nanmean(g)))
        gy[unknown] = 0; gx[unknown] = 0
        ny, nx = np.gradient(ndi.gaussian_filter(unknown.astype(float), 1.0))
        norm = np.hypot(nx, ny) + 1e-9
        iso_x, iso_y = -gy, gx
        # use the strongest isophote in the 3x3 neighbourhood (front pixels themselves are unknown)
        mag = np.hypot(iso_x, iso_y)
        mag_max = ndi.maximum_filter(mag, size=3)
        D = np.abs(iso_x[fy, fx] * nx[fy, fx] + iso_y[fy, fx] * ny[fy, fx]) / norm[fy, fx]
        D = np.maximum(D, 0.1 * mag_max[fy, fx]) / alpha + 1e-3
        k = int(np.argmax(C * D))
        py, px = int(np.clip(fy[k], r, h - r - 1)), int(np.clip(fx[k], r, w - r - 1))

        tgt = out[py - r:py + r + 1, px - r:px + r + 1]
        tmask = ~unknown[py - r:py + r + 1, px - r:px + r + 1]
        y0, y1 = max(r, py - search), min(h - r, py + search + 1)
        x0, x1 = max(r, px - search), min(w - r, px + search + 1)
        windows = np.lib.stride_tricks.sliding_window_view(out[y0 - r:y1 + r, x0 - r:x1 + r], (patch, patch), axis=(0, 1))
        # windows: (ny, nx, 3, patch, patch)
        diff = (windows - np.moveaxis(tgt, -1, 0)[None, None]) ** 2
        ssd = (diff * tmask[None, None, None]).sum(axis=(2, 3, 4))
        valid = source_ok[y0:y1, x0:x1]
        if not valid.any():  # widen to the whole image if the window has no source patch
            y0, y1, x0, x1 = r, h - r, r, w - r
            windows = np.lib.stride_tricks.sliding_window_view(out, (patch, patch), axis=(0, 1))
            diff = (windows - np.moveaxis(tgt, -1, 0)[None, None]) ** 2
            ssd = (diff * tmask[None, None, None]).sum(axis=(2, 3, 4))
            valid = source_ok[r:h - r, r:w - r]
        ssd = np.where(valid, ssd, np.inf)
        sy, sx = np.unravel_index(int(np.argmin(ssd)), ssd.shape)
        sy, sx = sy + y0, sx + x0
        src = out[sy - r:sy + r + 1, sx - r:sx + r + 1]
        fill = ~tmask
        tgt[fill] = src[fill]
        gray[py - r:py + r + 1, px - r:px + r + 1][fill] = src[fill].mean(axis=1)
        confidence[py - r:py + r + 1, px - r:px + r + 1][fill] = C[k]
        unknown[py - r:py + r + 1, px - r:px + r + 1] = False
    return out


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "exemplar") -> np.ndarray:
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; available: {sorted(METHODS)}")
    return METHODS[method](img, mask)
