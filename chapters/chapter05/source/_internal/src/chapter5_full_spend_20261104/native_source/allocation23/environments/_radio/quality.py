"""Terminal RGB SSIM for the fixed 64 by 64 image-transmission task.

The calculation matches ``skimage.metrics.structural_similarity`` 0.26 with
``SSIM_PARAMETERS`` below, using only NumPy and SciPy at runtime. Gaussian
statistics use an 11 by 11 window, population covariance, and a five-pixel
border crop; RGB channels are evaluated separately and averaged. SSIM is an
index in the theoretical range [-1, 1], not a percentage of pixels recovered.
Negative scores are retained without clipping.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .codec import IMAGE_SHAPE


SSIM_VERSION = "rgb-ssim-gaussian11-population-v1"
SSIM_PARAMETERS = {
    "data_range": 255,
    "channel_axis": -1,
    "gaussian_weights": True,
    "win_size": 11,
    "sigma": 1.5,
    "use_sample_covariance": False,
    "K1": 0.01,
    "K2": 0.03,
}


def _validate_images(images, *, batch: bool, name: str) -> np.ndarray:
    array = np.asarray(images)
    valid_shape = (
        array.ndim == 4 and array.shape[1:] == IMAGE_SHAPE
        if batch else array.shape == IMAGE_SHAPE
    )
    if not valid_shape or array.dtype != np.uint8:
        shape = "N,64,64,3" if batch else "64,64,3"
        raise ValueError(f"{name} must be uint8 RGB[{shape}]")
    return array


def _score_batch(sources: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    # NCHW keeps each channel's cropped spatial mean in the same order as the
    # reference's separate two-dimensional calculations. Never filter across
    # the batch or RGB axes.
    x = np.ascontiguousarray(sources.transpose(0, 3, 1, 2), dtype=np.float64)
    y = np.ascontiguousarray(reconstructions.transpose(0, 3, 1, 2), dtype=np.float64)

    def spatial_gaussian(values):
        return gaussian_filter(values, sigma=(0., 0., 1.5, 1.5),
                               truncate=3.5, mode="reflect")

    ux, uy = spatial_gaussian(x), spatial_gaussian(y)
    vx = spatial_gaussian(x * x) - ux * ux
    vy = spatial_gaussian(y * y) - uy * uy
    vxy = spatial_gaussian(x * y) - ux * uy
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a1, a2 = 2 * ux * uy + c1, 2 * vxy + c2
    b1, b2 = ux ** 2 + uy ** 2 + c1, vx + vy + c2
    score_map = (a1 * a2) / (b1 * b2)
    per_channel = score_map[:, :, 5:-5, 5:-5].mean(axis=(-2, -1), dtype=np.float64)
    return per_channel.mean(axis=1, dtype=np.float64)


def rgb_ssim(source, reconstruction) -> float:
    """Return RGB SSIM for two uint8 arrays of shape (64, 64, 3)."""
    source = _validate_images(source, batch=False, name="source")
    reconstruction = _validate_images(reconstruction, batch=False, name="reconstruction")
    return float(_score_batch(source[None], reconstruction[None])[0])


def batch_rgb_ssim(sources, reconstructions) -> np.ndarray:
    """Return one float64 SSIM per image in matching (N, 64, 64, 3) arrays.

    Empty batches return an empty float64 array. Internal chunks bound working
    memory independently of the number of terminals; inputs are never changed.
    """
    sources = _validate_images(sources, batch=True, name="sources")
    reconstructions = _validate_images(reconstructions, batch=True, name="reconstructions")
    if sources.shape != reconstructions.shape:
        raise ValueError("sources and reconstructions must have the same batch shape")
    scores = np.empty(len(sources), dtype=np.float64)
    for start in range(0, len(sources), 32):
        end = start + 32
        scores[start:end] = _score_batch(sources[start:end], reconstructions[start:end])
    return scores
