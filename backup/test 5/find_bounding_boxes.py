"""
Estrae box da maschera binaria (solo bordi) e usa la depth **solo** per lo scoring.
"""

from typing import List, Tuple
import cv2
import numpy as np

Box = Tuple[int, int, int, int]

# ---------------------------------------------------------------
def find_bounding_boxes(binary: np.ndarray) -> Tuple[List[Box], List]:
    """
    Restituisce (boxes, contours) lavorando su una binaria uint8.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        if ratio > 10 or ratio < 0.1:
            continue
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(binary.shape[1] - x, w + 2 * pad)
        h = min(binary.shape[0] - y, h + 2 * pad)
        boxes.append((x, y, w, h))
    return boxes, contours


def pick_best_box(boxes: List[Box], left_path: str, right_path: str):
    """
    Usa la depth map solo per la selezione finale.
    """
    if not boxes:
        return None

    from deapMap import relative_depth_map

    depth = relative_depth_map(left_path, right_path)

    # ricostruisco edge mask per il punteggio
    from edge_pipeline import edge_pipeline_full
    _, _, _, edge_mask = edge_pipeline_full(left_path)

    H, W = edge_mask.shape
    center = np.array([W // 2, H // 2])
    scores = []

    for (x, y, w, h) in boxes:
        area = w * h
        cx, cy = x + w // 2, y + h // 2
        dist_center = np.linalg.norm([cx - center[0], cy - center[1]])

        roi_edge = edge_mask[y:y + h, x:x + w]
        edge_density = np.sum(roi_edge > 0) / (area + 1e-6)

        roi_depth = depth[y:y + h, x:x + w]
        mean_depth = np.mean(roi_depth[roi_depth > 0]) if np.any(roi_depth > 0) else 0

        score = (
            0.15 * (area / (H * W)) +
            0.35 * mean_depth +
            0.35 * edge_density +
            0.15 * (1 - dist_center / np.sqrt(H ** 2 + W ** 2))
        )
        scores.append(score)

    return boxes[np.argmax(scores)]