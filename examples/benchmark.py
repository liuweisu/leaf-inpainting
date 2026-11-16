"""Benchmark all methods on synthetic damaged leaves. Run: python examples/benchmark.py"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from leafinpaint import data, methods, metrics

DOCS = Path(__file__).resolve().parent.parent / "docs"
METHOD_ORDER = ["harmonic", "biharmonic", "telea", "navier_stokes", "exemplar"]


def run(n_leaves: int = 8) -> None:
    results: dict[tuple[str, str], list[tuple[float, ...]]] = {}
    for kind in data.DAMAGE:
        for i in range(n_leaves):
            leaf = data.make_leaf(seed=100 + i)
            d = data.damage_leaf(leaf, kind, seed=200 + i)
            results.setdefault((kind, "no restoration"), []).append(
                (metrics.psnr(d.damaged, leaf.rgb, d.damage), metrics.ssim(d.damaged, leaf.rgb),
                 metrics.shape_iou(d.damaged, leaf.mask, d.damage), 0.0))
            for m in METHOD_ORDER:
                t = time.perf_counter()
                out = methods.inpaint(d.damaged, d.damage, m)
                dt = time.perf_counter() - t
                results.setdefault((kind, m), []).append(
                    (metrics.psnr(out, leaf.rgb, d.damage), metrics.ssim(out, leaf.rgb),
                     metrics.shape_iou(out, leaf.mask, d.damage), dt))

    print(f"Mean over {n_leaves} synthetic leaves (192x192) per damage type.\n")
    print("| damage | method | PSNR in hole (dB) | SSIM whole image | leaf shape IoU in hole | time (s) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for (kind, m), vals in results.items():
        a = np.mean(vals, axis=0)
        print(f"| {kind} | {m} | {a[0]:.1f} | {a[1]:.3f} | {a[2]:.3f} | {a[3]:.2f} |")


def figure() -> None:
    rows = []
    for kind, seed in (("bites", 3), ("holes", 4), ("tear", 5)):
        leaf = data.make_leaf(seed=seed)
        d = data.damage_leaf(leaf, kind, seed=seed)
        panels = [("damaged", d.damaged)] + [(m, methods.inpaint(d.damaged, d.damage, m)) for m in ("harmonic", "telea", "exemplar")]
        panels.append(("original", leaf.rgb))
        tiles = []
        for name, img in panels:
            im = Image.fromarray((img * 255).astype(np.uint8))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 6 + 6 * len(name), 14], fill=(0, 0, 0))
            dr.text((3, 1), name, fill=(255, 255, 255))
            tiles.append(np.asarray(im))
        rows.append(np.hstack(tiles))
        Image.fromarray((d.damaged * 255).astype(np.uint8)).save(DOCS / "samples" / f"damaged_{kind}.png")
        Image.fromarray((d.damage * 255).astype(np.uint8)).save(DOCS / "samples" / f"mask_{kind}.png")
    Image.fromarray(np.vstack(rows)).save(DOCS / "comparison.png")
    Image.fromarray((data.make_leaf(size=256, seed=11).rgb * 255).astype(np.uint8)).save(DOCS / "samples" / "leaf_intact.png")


if __name__ == "__main__":
    (DOCS / "samples").mkdir(parents=True, exist_ok=True)
    run()
    figure()
