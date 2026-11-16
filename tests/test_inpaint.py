import numpy as np
import pytest
from PIL import Image

from leafinpaint import cli, data, methods, metrics


@pytest.fixture(scope="module")
def case():
    leaf = data.make_leaf(size=128, seed=1)
    return leaf, data.damage_leaf(leaf, "bites", seed=2)


@pytest.mark.parametrize("method", sorted(methods.METHODS))
def test_known_pixels_unchanged_and_hole_improved(case, method):
    leaf, d = case
    out = methods.inpaint(d.damaged, d.damage, method)
    tol = 1 / 255 + 1e-9 if method in {"telea", "navier_stokes"} else 1e-9  # OpenCV works in 8 bit
    assert np.abs(out[~d.damage] - d.damaged[~d.damage]).max() <= tol
    assert metrics.psnr(out, leaf.rgb, d.damage) > metrics.psnr(d.damaged, leaf.rgb, d.damage) + 3


def test_harmonic_is_exact_for_linear_ramps():
    # a linear function is harmonic, so the Laplace fill must reproduce it exactly
    yy, xx = np.mgrid[0:40, 0:40] / 40
    img = np.dstack([xx, yy, 0.5 * xx + 0.25 * yy])
    mask = np.zeros((40, 40), bool); mask[10:30, 12:25] = True
    damaged = img.copy(); damaged[mask] = 0
    np.testing.assert_allclose(methods.harmonic(damaged, mask), img, atol=1e-6)


def test_exemplar_continues_texture_better_than_diffusion(case):
    leaf, d = case
    ex = metrics.shape_iou(methods.inpaint(d.damaged, d.damage, "exemplar"), leaf.mask, d.damage)
    hm = metrics.shape_iou(methods.inpaint(d.damaged, d.damage, "harmonic"), leaf.mask, d.damage)
    assert ex > hm


def test_global_ssim_hides_damage(case):
    """The point of the shape metric: whole-image SSIM barely notices a bitten leaf."""
    leaf, d = case
    assert metrics.ssim(d.damaged, leaf.rgb) > 0.9
    assert metrics.shape_iou(d.damaged, leaf.mask, d.damage) < 0.5


@pytest.mark.parametrize("kind", list(data.DAMAGE))
def test_damage_masks_touch_the_leaf(kind):
    leaf = data.make_leaf(size=128, seed=3)
    d = data.damage_leaf(leaf, kind, seed=3)
    assert d.damage.any() and (d.damage & leaf.mask).sum() > 0.5 * d.damage.sum()


def test_errors():
    img = np.zeros((10, 10, 3))
    with pytest.raises(ValueError):
        methods.inpaint(img, np.ones((10, 10), bool), "harmonic")
    with pytest.raises(KeyError):
        methods.inpaint(img, np.zeros((10, 10), bool), "magic")


def test_cli_roundtrip(tmp_path, case):
    _, d = case
    Image.fromarray((d.damaged * 255).astype(np.uint8)).save(tmp_path / "img.png")
    Image.fromarray((d.damage * 255).astype(np.uint8)).save(tmp_path / "mask.png")
    cli.main([str(tmp_path / "img.png"), str(tmp_path / "mask.png"), "-o", str(tmp_path / "out.png"), "--method", "telea"])
    assert Image.open(tmp_path / "out.png").size == (128, 128)
