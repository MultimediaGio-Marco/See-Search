"""
edge_pipeline.py  –  versione “poco più flessibile”
Nessun modello IA, solo parametri esposti.
"""

import cv2
import numpy as np
import pywt
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label
from skimage.morphology import remove_small_objects
from skimage.color import rgb2lab
from PIL import Image

# ---------- funzioni interne ----------
def sobel_custom(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return dx, dy

def wavelet_denoise(image, level=2):
    coeffs = pywt.wavedec2(image, 'db1', level=level)
    threshold = 0.04 * np.max(image)
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [
        tuple(pywt.threshold(c, threshold, mode='soft') for c in detail)
        for detail in coeffs[1:]
    ]
    return pywt.waverec2(coeffs_thresh, 'db1')

def color_based_segmentation(image_rgb, a_weight=1.0, b_weight=1.0):
    lab = rgb2lab(image_rgb)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    a_norm = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, a_thr = cv2.threshold(a_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b_thr = cv2.threshold(b_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # pesatura semplice
    color_mask = cv2.bitwise_or(
        (a_thr * a_weight).astype(np.uint8),
        (b_thr * b_weight).astype(np.uint8)
    )
    return color_mask

def morphological_cleanup(binary_image,
                          close_ks=5,
                          open_ks=3,
                          fill_holes=True):
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))

    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, k_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)
    if fill_holes:
        filled = binary_fill_holes(opened).astype(np.uint8) * 255
    else:
        filled = opened
    return filled

# ---------- pipeline principale ----------
def edge_pipeline_py(image_path, flex=None):
    """
    flex: dict con chiavi opzionali
        a_weight, b_weight,
        close_ks, open_ks, fill_holes,
        min_area, max_aspect, nms_iou, padding
    """
    flex = flex or {}

    # 0. load
    img_rgb = np.array(Image.open(image_path).convert('RGB'))

    # 1. Y channel
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y = ycbcr[:, :, 0].astype(np.float64)

    # 2. color
    color_mask = color_based_segmentation(
        img_rgb,
        a_weight=flex.get('a_weight', 1.0),
        b_weight=flex.get('b_weight', 1.0)
    )

    # 3. edge detection
    Y_blur = cv2.GaussianBlur(Y, (5, 5), 1)
    Y_bil  = cv2.bilateralFilter(Y_blur.astype(np.uint8), 9, 75, 75).astype(np.float64)

    dx, dy = sobel_custom(Y_bil)
    lap = cv2.Laplacian(Y_bil, cv2.CV_64F)
    canny = cv2.Canny(Y_bil.astype(np.uint8), 50, 150)

    magnitude = np.sqrt(dx**2 + dy**2) + np.abs(lap)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    combined = cv2.bitwise_or(magnitude, canny)
    combined = cv2.bitwise_or(combined, color_mask)

    # 4. denoise
    denoised = wavelet_denoise(combined, level=2)
    denoised_u8 = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5. binarizza
    _, BW = cv2.threshold(denoised_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. morfologia + small object removal
    BW = morphological_cleanup(
        BW,
        close_ks=flex.get('close_ks', 5),
        open_ks=flex.get('open_ks', 3),
        fill_holes=flex.get('fill_holes', True)
    )
    BW_clean = remove_small_objects(BW > 0, min_size=flex.get('min_area', 200))
    BW_final = (BW_clean * 255).astype(np.uint8)

    return BW_final, combined, denoised_u8