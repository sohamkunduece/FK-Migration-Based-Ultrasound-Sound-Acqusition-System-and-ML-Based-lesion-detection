
import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import denoise_bilateral


def enhancement2(bmode):

    # ========================================================
    # 1. NORMALIZE INPUT
    # MATLAB: img = mat2gray(bmode)
    # ========================================================

    bmode = np.asarray(bmode, dtype=np.float64)

    img_min = np.min(bmode)
    img_max = np.max(bmode)

    if img_max == img_min:
        img = np.zeros_like(bmode)
    else:
        img = (
            (bmode - img_min)
            / (img_max - img_min)
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

    img = denoise_bilateral(
        img,
        win_size=5,
        sigma_color=0.1,
        sigma_spatial=1.0,
        bins=10000,
        mode='constant',
        cval=0,
        channel_axis=None
    )

    # ========================================================
    # 4. CONVERT TO dB
    # MATLAB: 20*log10(img+eps)
    # ========================================================

    enhanced_img = (
        20 * np.log10(
            img + np.finfo(np.float64).eps
        )
    )

    return enhanced_img
