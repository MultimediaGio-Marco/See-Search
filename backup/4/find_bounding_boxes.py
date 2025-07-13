"""
find_boxes.py  –  parametrizzato
"""

import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# ---------- watershed ----------
def watershed_segmentation(binary_image,
                           min_distance=20,
                           rel_thr=0.3):
    dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    if dist.max() < 2:               # troppo piccolo
        return None
    peaks = peak_local_max(dist,
                           min_distance=min_distance,
                           threshold_abs=rel_thr * dist.max())
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks):
        markers[r, c] = i + 1
    labels = watershed(-dist, markers, mask=binary_image)
    return labels

# ---------- NMS ----------
def non_max_suppression(boxes, overlap_threshold=0.3):
    if len(boxes) == 0:
        return []
    boxes_xyxy = np.array([(x, y, x+w, y+h) for x, y, w, h in boxes])
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    idxs = np.argsort(areas)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[idxs[1:], 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[idxs[1:], 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[idxs[1:], 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[idxs[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs[1:]] - inter)
        idxs = idxs[1:][iou < overlap_threshold]
    return [(int(x1), int(y1), int(x2-x1), int(y2-y1))
            for x1, y1, x2, y2 in boxes_xyxy[keep]]

# ---------- main ----------
def find_bounding_boxes(binary_image,
                        min_area=200,
                        max_aspect=10,
                        padding=5,
                        nms_iou=0.3):
    labels = watershed_segmentation(binary_image)
    boxes, contours = [], []

    if labels is not None:
        for v in np.unique(labels):
            if v == 0:
                continue
            mask = (labels == v).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cnts)
    else:
        cnts, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cnts)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h
        if aspect > max_aspect or aspect < 1/max_aspect:
            continue
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary_image.shape[1] - x, w + 2*padding)
        h = min(binary_image.shape[0] - y, h + 2*padding)
        boxes.append((x, y, w, h))

    final = non_max_suppression(boxes, overlap_threshold=nms_iou)
    if not final:
        return None, []
    largest = max(final, key=lambda b: b[2]*b[3])
    return largest, contours