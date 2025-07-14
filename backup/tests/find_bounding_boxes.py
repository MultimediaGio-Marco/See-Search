import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt


def improved_watershed_segmentation(binary_image, min_distance=20, threshold_rel=0.3):
    """Watershed su -distance_transform con peak_local_max per separare oggetti contigui"""
    # 1) Calcola distance transform
    dist = distance_transform_edt(binary_image)
    # 2) Trova i picchi locali sul dist negativo
    coords = peak_local_max(dist,
                             min_distance=min_distance,
                             threshold_rel=threshold_rel,
                             labels=binary_image)
    # 3) Crea marker dalla lista di picchi
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    # 4) Applica watershed sul segno negativo
    labels = watershed(-dist, markers, mask=binary_image)
    return labels, len(coords)


def find_bounding_boxes(binary_image):
    """Produce una bounding box per ogni mela utilizzando watershed migliorato"""
    # Esegui watershed per separare mele contigue
    labels, n_markers = improved_watershed_segmentation(
        binary_image,
        min_distance=30,
        threshold_rel=0.2
    )
    boxes = []
    # Per ogni label estrai il box
    for lbl in range(1, n_markers+1):
        mask = (labels == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        # Prendi il contorno più grande
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Padding controllato
        pad = int(0.05 * max(w, h))
        x, y = max(0, x-pad), max(0, y-pad)
        w = min(binary_image.shape[1]-x, w+2*pad)
        h = min(binary_image.shape[0]-y, h+2*pad)
        boxes.append((x, y, w, h))
    return boxes, labels


def visualize_intermediate_results(binary_image, labels=None):
    """Visualizza overlay dei segmenti watershed"""
    if labels is None:
        return cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    # Color map semplice
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        mask = labels == lbl
        color = tuple(np.random.randint(0,255,3).tolist())
        out[mask] = color
    return out
