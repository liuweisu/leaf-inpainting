"""Adapters for deep inpainting models (optional; needs a GPU in practice).

Classical methods only see the pixels around the hole. Generative models
(LaMa, Stable Diffusion inpainting, BrushNet) have learned what leaves look
like, which helps with large holes and also makes it easy for them to invent
a plausible but wrong leaf outline. The shape metric in ``metrics.py`` is
meant to catch exactly that.

``pip install diffusers transformers accelerate torch``, then::

    from leafinpaint import deep
    deep.register_stable_diffusion()           # adds "sd_inpaint" to methods.METHODS
    out = methods.inpaint(img, mask, "sd_inpaint")

Not exercised by the test suite or CI (no GPU there).
"""

from __future__ import annotations

import numpy as np

from .methods import register


def register_stable_diffusion(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
    prompt: str = "a photo of a healthy green leaf on white paper, botanical scan",
    steps: int = 30,
    seed: int = 0,
    device: str | None = None,
) -> None:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

    @register("sd_inpaint")
    def sd_inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        image = Image.fromarray((img * 255).astype(np.uint8)).resize((512, 512))
        m = Image.fromarray((mask * 255).astype(np.uint8)).resize((512, 512))
        result = pipe(prompt=prompt, image=image, mask_image=m, num_inference_steps=steps,
                      generator=torch.Generator(device).manual_seed(seed)).images[0]
        out = np.asarray(result.resize((w, h))).astype(float) / 255
        # keep known pixels exactly; the VAE round trip changes them slightly
        return np.where(mask[..., None], out, img)
