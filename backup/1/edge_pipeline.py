import cv2
import numpy as np
import pywt
from scipy.ndimage import binary_closing, binary_opening
from scipy.ndimage import label
from skimage.morphology import remove_small_objects, disk, binary_closing as sk_binary_closing
from skimage.color import rgb2ycbcr
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

def edge_pipeline_py(image_path):
    # [0] Carica immagine
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # [1] RGB → YCbCr
    ycbcr = rgb2ycbcr(image_np)

    # [2] Estrai Y
    Y = ycbcr[:, :, 0].astype(np.float64)

    # [3] Gaussian Blur
    Y_blur = cv2.GaussianBlur(Y, (5, 5), sigmaX=1)

    # [4] Sobel + Laplaciano
    dx, dy = sobel_custom(Y_blur)
    laplacian = cv2.Laplacian(Y_blur, cv2.CV_64F)

    # [5] Magnitudine
    magnitude = np.sqrt(dx**2 + dy**2) + np.abs(laplacian)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # [6] Wavelet denoise
    denoised = wavelet_denoise(magnitude, level=2)

    # [7] Threshold
    BW = denoised > 40

    # [8] Pulizia morfologica
    BW = sk_binary_closing(BW, footprint=disk(1))
    BW = remove_small_objects(BW, min_size=50)

    return BW.astype(np.uint8) * 255, magnitude.astype(np.uint8), denoised.astype(np.uint8)
