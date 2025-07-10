import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.color import rgb2ycbcr, rgb2lab
from PIL import Image

def sobel_custom(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return dx, dy

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

def morphological_cleanup(binary_image, close_kernel_size=5, open_kernel_size=3):
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    filled = binary_fill_holes(opened).astype(np.uint8) * 255
    return filled

def edge_pipeline_py(image_path):
    # [0] Carica immagine
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # [1] Estrai canale Y
    ycbcr = rgb2ycbcr(image_np)
    Y = ycbcr[:, :, 0].astype(np.float64)

    # [2] Segmentazione colore
    color_mask = color_based_segmentation(image_np)

    # [3] Edge detection
    Y_blur = cv2.GaussianBlur(Y, (5, 5), sigmaX=1)

    dx, dy = sobel_custom(Y_blur)
    laplacian = cv2.Laplacian(Y_blur, cv2.CV_64F)
    canny = cv2.Canny(Y_blur.astype(np.uint8), 50, 150)

    magnitude = np.sqrt(dx**2 + dy**2) + np.abs(laplacian)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    combined_edges = cv2.bitwise_or(magnitude.astype(np.uint8), canny)
    combined_edges = cv2.bitwise_or(combined_edges, color_mask)

    # [4] Soglia (senza wavelet)
    denoised_norm = combined_edges  # usa direttamente i bordi combinati
    _, BW = cv2.threshold(denoised_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # [5] Cleanup morfologico
    BW = morphological_cleanup(BW)

    # [6] Rimuovi oggetti piccoli
    BW_bool = BW > 0
    BW_cleaned = remove_small_objects(BW_bool, min_size=200)
    BW_final = (BW_cleaned * 255).astype(np.uint8)

    return BW_final, combined_edges, denoised_norm
