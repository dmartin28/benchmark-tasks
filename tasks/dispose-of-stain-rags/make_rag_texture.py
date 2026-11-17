"""Generate rag.png: an off-white cotton rag blotched unevenly with dark walnut stain.

Run once; the PNGs are committed. The look of a used stain rag: bare cotton showing between soaked
patches, the stain darkest where it pooled, darker "tide lines" where wet patches dried at their edges,
and a faint weave. Pure numpy + Pillow.
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
N = 512                                   # pixels across the 250 mm rag (flat)
COTTON = np.array((178, 166, 142), float)  # sRGB, grubby unbleached cotton
STAIN = np.array((48, 22, 8), float)       # sRGB, dark walnut where soaked
EDGE = np.array((40, 18, 6), float)        # dried rims of the wet patches


def _smooth_noise(rng, n, cells):
    """Value noise: a coarse random grid, bicubically upsampled (Pillow), in 0..1."""
    g = rng.random((cells, cells)).astype(np.float32)
    img = Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
    return np.asarray(img, float) / 255.0


def make(seed, out):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N] / N
    # Wetness: a few big soaked blobs where the rag was pressed into the stain, plus mottling.
    wet = np.zeros((N, N))
    for _ in range(rng.integers(4, 7)):
        cx, cy = rng.uniform(0.1, 0.9, 2)
        sx, sy = rng.uniform(0.08, 0.22, 2)
        wet += rng.uniform(0.6, 1.0) * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    wet += 0.55 * _smooth_noise(rng, N, 6) + 0.35 * _smooth_noise(rng, N, 14) + 0.15 * _smooth_noise(rng, N, 40)
    wet = (wet - wet.min()) / (wet.max() - wet.min())
    level = rng.uniform(0.30, 0.36)        # well over half of the cloth soaked
    soak = np.clip((wet - level) / 0.10, 0.0, 1.0)            # 0 bare cotton .. 1 soaked
    depth = np.clip((wet - level) / (1.0 - level), 0.0, 1.0)  # darker toward the middle of a patch
    tide = 0.6 * np.exp(-((wet - level - 0.015) / 0.02) ** 2)         # dark rim at each patch's edge
    faint = 0.25 * np.clip((wet - level + 0.12) / 0.12, 0.0, 1.0) * (1 - soak)   # light halo of wicked stain
    k = np.clip(0.55 * soak + 0.45 * soak * depth + faint, 0.0, 1.0)[..., None]
    rgb = COTTON * (1 - k) + STAIN * k
    rgb = rgb * (1 - 0.5 * tide[..., None]) + EDGE * 0.5 * tide[..., None]
    # Faint plain weave and fibre grain.
    weave = 0.04 * (np.sin(2 * np.pi * x * N / 4) * np.sin(2 * np.pi * y * N / 4))
    rgb *= (1 + weave + 0.05 * (rng.random((N, N)) - 0.5))[..., None]
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out)
    return out


if __name__ == "__main__":
    print(make(29, HERE / "rag.png"))
