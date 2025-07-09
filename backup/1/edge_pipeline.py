import cv2
import numpy as np
import pywt
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, disk, binary_closing as sk_binary_closing
from skimage.color import rgb2ycbcr, rgb2lab
from skimage.util import img_as_float
from PIL import Image


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


def color_based_segmentation(image_rgb):
    lab = rgb2lab(image_rgb)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    a_norm = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b_norm = cv2.normalize(b_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, a_thresh = cv2.threshold(a_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b_thresh = cv2.threshold(b_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    color_mask = cv2.bitwise_or(a_thresh, b_thresh)
    return color_mask


def morphological_cleanup(binary_image, close_kernel_size=3, open_kernel_size=2):
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    filled = binary_fill_holes(opened).astype(np.uint8) * 255
    return filled


def edge_pipeline_py(image_path, mode="otsu"):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    ycbcr = rgb2ycbcr(image_np)
    Y = ycbcr[:, :, 0].astype(np.float64)

    color_mask = color_based_segmentation(image_np)
    Y_blur = cv2.GaussianBlur(Y, (5, 5), sigmaX=1)

    dx, dy = sobel_custom(Y_blur)
    laplacian = cv2.Laplacian(Y_blur, cv2.CV_64F)
    cost = np.sqrt(dx**2 + dy**2) + np.abs(laplacian)
    cost_norm = cv2.normalize(cost, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if mode == "otsu":
        # [1] Segmentazione con soglia di Otsu sulla mappa di costo
        _, edge_mask = cv2.threshold(cost_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # [2] Segmentazione con soglia fissa (fallback)
        edge_mask = (cost_norm > 40).astype(np.uint8) * 255

    # [3] Fusione con segmentazione a colori
    combined_mask = cv2.bitwise_or(edge_mask, color_mask)

    # [4] Pulizia finale
    cleaned = morphological_cleanup(combined_mask, close_kernel_size=5, open_kernel_size=3)
    BW_bool = cleaned > 0
    BW_cleaned = remove_small_objects(BW_bool, min_size=200)
    BW_final = (BW_cleaned * 255).astype(np.uint8)

    return BW_final, cost_norm, cost_norm
