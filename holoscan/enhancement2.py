
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import median_filter


def bilateral_filter_gpu(
    img,
    win_size=5,
    sigma_color=0.1,
    sigma_spatial=1.0
):
    """
    GPU bilateral filter using CuPy.

    Input and output remain on the GPU.
    Uses constant-zero boundary padding.
    """

    img = cp.asarray(img, dtype=cp.float64)

    if img.ndim != 2:
        raise ValueError("Input image must be 2D.")

    radius = win_size // 2

    height, width = img.shape

    # Constant-zero padding
    padded = cp.pad(
        img,
        ((radius, radius), (radius, radius)),
        mode='constant',
        constant_values=0
    )

    # Coordinate grid for spatial Gaussian weights
    coords = cp.arange(-radius, radius + 1)

    yy, xx = cp.meshgrid(coords, coords, indexing='ij')

    spatial_weights = cp.exp(
        -(xx**2 + yy**2) /
        (2.0 * sigma_spatial**2)
    )

    # Accumulation arrays
    weighted_sum = cp.zeros_like(img)
    weight_sum = cp.zeros_like(img)

    # Center pixel
    center = img

    # Bilateral filtering
    for dy in range(-radius, radius + 1):

        for dx in range(-radius, radius + 1):

            neighbor = padded[
                radius + dy:radius + dy + height,
                radius + dx:radius + dx + width
            ]

            spatial_weight = spatial_weights[
                dy + radius,
                dx + radius
            ]

            range_weight = cp.exp(
                -((neighbor - center) ** 2) /
                (2.0 * sigma_color**2)
            )

            weight = spatial_weight * range_weight

            weighted_sum += weight * neighbor
            weight_sum += weight

    # Normalize
    result = weighted_sum / cp.maximum(
        weight_sum,
        cp.finfo(cp.float64).eps
    )

    return result


def enhancement2(bmode):

    """
    CUDA-optimized enhancement2.

    Input:
        bmode: 2D NumPy or CuPy array

    Output:
        Enhanced 2D CuPy GPU array in dB.
    """

    # ========================================================
    # 1. MOVE INPUT TO GPU AND NORMALIZE
    # MATLAB: img = mat2gray(bmode)
    # ========================================================

    bmode = cp.asarray(bmode, dtype=cp.float64)

    if bmode.ndim != 2:
        raise ValueError("bmode must be a 2D image.")

    img_min = cp.min(bmode)
    img_max = cp.max(bmode)

    img = cp.where(
        img_max == img_min,
        cp.zeros_like(bmode),
        (bmode - img_min) /
        cp.maximum(
            img_max - img_min,
            cp.finfo(cp.float64).eps
        )
    )

    # ========================================================
    # 2. MEDIAN FILTER
    # MATLAB: medfilt2(img,[3 3])
    # ========================================================

    img = median_filter(
        img,
        size=(3, 3),
        mode='constant',
        cval=0
    )

    # ========================================================
    # 3. BILATERAL FILTER
    # MATLAB: imbilatfilt(img)
    # ========================================================

    img = bilateral_filter_gpu(
        img,
        win_size=5,
        sigma_color=0.1,
        sigma_spatial=1.0
    )

    # ========================================================
    # 4. CONVERT TO dB
    # MATLAB: 20*log10(img+eps)
    # ========================================================

    enhanced_img = (
        20.0 * cp.log10(
            img + cp.finfo(cp.float64).eps
        )
    )

    return enhanced_img
