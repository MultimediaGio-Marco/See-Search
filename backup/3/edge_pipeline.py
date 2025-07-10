import cv2
import numpy as np
import pywt
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes
from scipy.ndimage import label, generate_binary_structure
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
    # thresholding solo sui dettagli (non approssimazione)
    threshold = 0.04 * np.max(image)
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [
        tuple(pywt.threshold(c, threshold, mode='soft') for c in detail)
        for detail in coeffs[1:]
    ]
    return pywt.waverec2(coeffs_thresh, 'db1')

def color_based_segmentation(image_rgb):
    """Use color information for better segmentation"""
    # Convert to LAB color space (better for object separation)
    lab = rgb2lab(image_rgb)
    
    # Use A and B channels for color segmentation
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    # Normalize channels
    a_norm = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b_norm = cv2.normalize(b_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply Otsu thresholding to color channels
    _, a_thresh = cv2.threshold(a_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b_thresh = cv2.threshold(b_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine color information
    color_mask = cv2.bitwise_or(a_thresh, b_thresh)
    
    return color_mask

def morphological_cleanup(binary_image, close_kernel_size=3, open_kernel_size=2):
    """Advanced morphological operations for cleanup"""
    # Create kernels
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    
    # Close gaps
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, close_kernel)
    
    # Remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    
    # Fill holes
    filled = binary_fill_holes(opened).astype(np.uint8) * 255
    
    return filled

def edge_pipeline_py(image_path):
    """Improved edge detection pipeline - compatible with your main.py"""
    # [0] Carica immagine
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # [1] RGB → YCbCr (keep original approach)
    ycbcr = rgb2ycbcr(image_np)
    Y = ycbcr[:, :, 0].astype(np.float64)

    # [2] Add color-based segmentation
    color_mask = color_based_segmentation(image_np)

    # [3] Improved edge detection
    # Original approach with improvements
    Y_blur = cv2.GaussianBlur(Y, (5, 5), sigmaX=1)
    
    # Apply bilateral filter to reduce noise
    Y_bilateral = cv2.bilateralFilter(Y_blur.astype(np.uint8), 9, 75, 75).astype(np.float64)
    
    # [4] Multi-scale edge detection
    dx, dy = sobel_custom(Y_bilateral)
    laplacian = cv2.Laplacian(Y_bilateral, cv2.CV_64F)
    
    # Additional Canny edges
    canny = cv2.Canny(Y_bilateral.astype(np.uint8), 50, 150)
    
    # [5] Combine edge information
    magnitude = np.sqrt(dx**2 + dy**2) + np.abs(laplacian)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Combine with Canny and color information
    combined_edges = cv2.bitwise_or(magnitude.astype(np.uint8), canny)
    combined_edges = cv2.bitwise_or(combined_edges, color_mask)

    # [6] Wavelet denoise
    denoised = wavelet_denoise(combined_edges, level=2)

    # [7] Improved threshold - use Otsu method
    denoised_norm = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, BW = cv2.threshold(denoised_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # [8] Improved morphological operations
    BW = morphological_cleanup(BW, close_kernel_size=5, open_kernel_size=3)
    
    # [9] Remove small objects
    BW_bool = BW > 0
    BW_cleaned = remove_small_objects(BW_bool, min_size=200)
    BW_final = (BW_cleaned * 255).astype(np.uint8)

    return BW_final, combined_edges, denoised_norm