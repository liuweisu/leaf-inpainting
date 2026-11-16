"""leafinpaint IMAGE MASK -o OUT [--method exemplar]

MASK is an image where white (or any non-zero) pixels mark the region to restore.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from . import __version__, methods


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="leafinpaint", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("mask")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--method", default="exemplar", choices=sorted(methods.METHODS))
    p.add_argument("--version", action="version", version=__version__)
    args = p.parse_args(argv)

    img = np.asarray(Image.open(args.image).convert("RGB")).astype(float) / 255
    mask = np.asarray(Image.open(args.mask).convert("L")) > 127
    if mask.shape != img.shape[:2]:
        raise SystemExit(f"mask size {mask.shape[::-1]} differs from image size {img.shape[1::-1]}")
    out = methods.inpaint(img, mask, args.method)
    Image.fromarray((out * 255).round().astype(np.uint8)).save(args.out)
    print(f"restored {mask.sum()} pixels with {args.method} -> {args.out}")


if __name__ == "__main__":
    main()
