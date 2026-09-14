"""Colorization quality metrics."""

import numpy as np
from skimage import color, metrics


def _as_rgb_float(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Images must be RGB arrays with shape (height, width, 3)")

    image = image.astype(np.float32, copy=False)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def evaluate_colorization(original_rgb, colorized_rgb):
    """Compare two aligned RGB images using several image-quality metrics.

    CIEDE2000 and MSE are error metrics where lower is better. PSNR and SSIM
    are quality metrics where higher is better. The reported similarity score
    is a convenient 0-100 presentation metric:
    an exact match scores 100, and every unit of mean CIEDE2000 error reduces
    the score by one point. Scores are clamped at zero.
    """
    original = _as_rgb_float(original_rgb)
    colorized = _as_rgb_float(colorized_rgb)
    if original.shape != colorized.shape:
        raise ValueError("Original and colorized images must have the same dimensions")

    original_lab = color.rgb2lab(original)
    colorized_lab = color.rgb2lab(colorized)
    delta_e = color.deltaE_ciede2000(original_lab, colorized_lab)
    mean_delta_e = float(np.mean(delta_e))
    mse = float(np.mean(np.square(original - colorized)))
    psnr = float("inf") if mse == 0.0 else float(metrics.peak_signal_noise_ratio(
        original, colorized, data_range=1.0
    ))
    if min(original.shape[:2]) < 3:
        raise ValueError("SSIM requires images with both dimensions at least 3 pixels")
    win_size = min(7, min(original.shape[:2]))
    if win_size % 2 == 0:
        win_size -= 1
    ssim = float(metrics.structural_similarity(
        original,
        colorized,
        channel_axis=-1,
        data_range=1.0,
        win_size=win_size,
    ))

    return {
        "mean_delta_e": round(mean_delta_e, 4),
        "similarity_score": round(max(0.0, 100.0 - mean_delta_e), 2),
        "mse": round(mse, 8),
        "psnr": None if np.isinf(psnr) else round(psnr, 4),
        "ssim": round(ssim, 4),
    }
