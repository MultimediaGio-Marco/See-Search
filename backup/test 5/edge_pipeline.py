"""
Edge detection **classico** (luminance + colori + wavelet + morfologia).
Restituisce 4 array uint8: originale, canny+magnitude, wavelet, binaria.
"""

import cv2
import numpy as np
import pywt
from PIL import Image
from skimage.color import rgb2ycbcr, rgb2lab
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

BLUR_K   = (5, 5)
BILAT    = (9, 75, 75)
CANNY_TH = (50, 150)
MORPH_KC = 5
MORPH_KO = 3
MIN_PX   = 200


# ------------------------------------------------------------------
def _wavelet_denoise(img, level=2):
    coeffs = pywt.wavedec2(img, 'db1', level=level)
    thr = 0.04 * img.max()
    coeffs[1:] = [tuple(pywt.threshold(c, thr, mode='soft') for c in detail)
                  for detail in coeffs[1:]]
    return pywt.waverec2(coeffs, 'db1')


def _color_mask(rgb):
    lab = rgb2lab(rgb)
    a = cv2.normalize(lab[:, :, 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b = cv2.normalize(lab[:, :, 2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, a_bin = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b_bin = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_or(a_bin, b_bin)


def edge_pipeline_full(image_path: str):
    """
    Ritorna:
    0) immagine originale BGR
    1) canny+magnitude
    2) dopo wavelet
    3) binaria finale
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- luminanza ---
    ycbcr = rgb2ycbcr(img_rgb)
    Y = ycbcr[:, :, 0].astype(np.float32)
    Y = cv2.GaussianBlur(Y, BLUR_K, 1)
    Y = cv2.bilateralFilter(Y.astype(np.uint8), *BILAT).astype(np.float32)

    dx = cv2.Sobel(Y, cv2.CV_64F, 1, 0, 3)
    dy = cv2.Sobel(Y, cv2.CV_64F, 0, 1, 3)
    mag = np.sqrt(dx ** 2 + dy ** 2)

    canny = cv2.Canny(Y.astype(np.uint8), *CANNY_TH)
    combined = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    combined = cv2.bitwise_or(combined, canny)

    # --- colori ---
    color_edges = _color_mask(img_rgb)
    combined = cv2.bitwise_or(combined, color_edges)

    # --- wavelet denoise ---
    denoised = _wavelet_denoise(combined).astype(np.uint8)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- morfologia finale ---
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KC, MORPH_KC))
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KO, MORPH_KO))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_c)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_o)
    binary = binary_fill_holes(binary).astype(np.uint8) * 255
    binary = remove_small_objects(binary.astype(bool), MIN_PX)
    binary = (binary * 255).astype(np.uint8)

    return img_bgr, combined, denoised, binary