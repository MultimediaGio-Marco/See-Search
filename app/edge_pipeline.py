import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from PIL import Image

def adaptive_color_segmentation(image_rgb, method='kmeans', n_clusters=5):
    """Segmentazione colore adattiva per oggetti generici"""
    if method == 'kmeans':
        # K-means clustering per segmentare colori dominanti
        data = image_rgb.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Ricostruisci l'immagine segmentata
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image_rgb.shape)
        
        return segmented_image
    
    elif method == 'threshold':
        # Sogliatura adattiva su più canali
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        return adaptive_thresh
    
    return None

def multi_scale_edge_detection(image_gray, scales=[1.0, 1.5, 2.0]):
    """Edge detection multi-scala per catturare dettagli a diverse dimensioni"""
    edges_combined = np.zeros_like(image_gray)
    
    for scale in scales:
        # Blur con scala variabile
        ksize = int(3 * scale)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(image_gray, (ksize, ksize), scale)
        
        # Canny con soglie adattive
        mean_intensity = np.mean(blurred)
        std_intensity = np.std(blurred)
        
        lower_thresh = max(30, int(mean_intensity - 0.5 * std_intensity))
        upper_thresh = min(255, int(mean_intensity + 0.5 * std_intensity))
        
        edges = cv2.Canny(blurred, lower_thresh, upper_thresh)
        edges_combined = cv2.bitwise_or(edges_combined, edges)
    
    return edges_combined

def gradient_magnitude_segmentation(image_gray):
    """Segmentazione basata su gradiente per bordi oggetti"""
    # Calcola gradienti Sobel
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitudine del gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))
    
    # Soglia adattiva sulla magnitudine
    thresh_val = np.mean(magnitude) + 0.5 * np.std(magnitude)
    _, binary = cv2.threshold(magnitude, thresh_val, 255, cv2.THRESH_BINARY)
    
    return binary

def generic_object_pipeline(image_path, segmentation_method='combined', min_object_size=1000):
    """Pipeline generico per segmentazione di qualsiasi oggetto"""
    # Carica immagine
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    if segmentation_method == 'edges':
        # Solo edge detection
        mask = multi_scale_edge_detection(gray)
        
    elif segmentation_method == 'gradient':
        # Solo gradiente
        mask = gradient_magnitude_segmentation(gray)
        
    elif segmentation_method == 'color':
        # Solo segmentazione colore
        segmented = adaptive_color_segmentation(image_np, method='kmeans')
        gray_seg = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray_seg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif segmentation_method == 'combined':
        # Combinazione di tutti i metodi
        edges = multi_scale_edge_detection(gray)
        gradient_mask = gradient_magnitude_segmentation(gray)
        
        # Combina edge e gradiente
        combined = cv2.bitwise_or(edges, gradient_mask)
        
        # Aggiungi informazioni di colore
        segmented_color = adaptive_color_segmentation(image_np, method='kmeans')
        gray_seg = cv2.cvtColor(segmented_color, cv2.COLOR_RGB2GRAY)
        _, color_mask = cv2.threshold(gray_seg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combina con peso
        mask = cv2.addWeighted(combined, 0.7, color_mask, 0.3, 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Pulizia morfologica generica
    mask = morphological_cleanup_generic(mask)
    
    # Rimozione oggetti piccoli
    mask_bool = mask > 0
    mask_cleaned = remove_small_objects(mask_bool, min_size=min_object_size)
    final_mask = (mask_cleaned * 255).astype(np.uint8)
    
    return final_mask

def morphological_cleanup_generic(binary_image, kernel_size=5):
    """Pulizia morfologica generica per qualsiasi forma"""
    # Kernel adattivo basato sulla dimensione dell'immagine
    h, w = binary_image.shape
    adaptive_kernel_size = max(3, min(kernel_size, min(h, w) // 50))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (adaptive_kernel_size, adaptive_kernel_size))
    
    # Operazioni morfologiche
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Riempi buchi
    filled = binary_fill_holes(opened).astype(np.uint8) * 255
    
    return filled

def watershed_generic(binary_image, min_distance=20):
    """Watershed generico per separare oggetti sovrapposti"""
    # Distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    # Trova picchi locali adattativamente
    # Usa una soglia relativa invece di assoluta
    threshold = 0.3 * np.max(dist_transform)
    _, sure_fg = cv2.threshold(dist_transform, threshold, 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Trova markers usando peak_local_max per maggiore controllo
    coordinates = peak_local_max(dist_transform, min_distance=min_distance, 
                                   threshold_abs=threshold)
    
    # Crea markers
    markers = np.zeros_like(binary_image, dtype=np.int32)
    for i, coord in enumerate(coordinates):
        markers[coord[0], coord[1]] = i + 1
    
    # Background
    sure_bg = cv2.dilate(binary_image, np.ones((3,3), np.uint8), iterations=2)
    
    # Applica watershed
    if len(coordinates) > 0:
        image_3ch = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        markers = watershed(-dist_transform, markers, mask=binary_image)
        
        # Genera segmentazione finale
        segmentation = np.zeros_like(binary_image)
        segmentation[markers > 0] = 255
        
        return segmentation
    else:
        return binary_image

def detect_and_segment_objects(image_path, method='combined', apply_watershed=True):
    """Funzione principale per rilevare e segmentare oggetti generici"""
    # Segmentazione iniziale
    mask = generic_object_pipeline(image_path, segmentation_method=method)
    
    if np.sum(mask) > 0:
        # Applica watershed se richiesto
        if apply_watershed:
            segmented = watershed_generic(mask)
        else:
            segmented = mask
            
        # Trova contorni per contare oggetti
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contorni troppo piccoli
        min_area = 500
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        print(f"Oggetti rilevati: {len(valid_contours)}")
        print(f"Pixel attivi nella segmentazione: {np.sum(segmented > 0)}")
        
        return segmented, valid_contours
    else:
        print("Nessun oggetto rilevato con i parametri attuali")
        return None, []
