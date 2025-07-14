import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from edge_pipeline import edge_pipeline_py
from deapMap import relative_depth_map

def watershed_segmentation(binary_image, min_distance=20):
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    local_maxima = peak_local_max(dist_transform, min_distance=min_distance, 
                                  threshold_abs=0.3 * dist_transform.max())

    markers = np.zeros(dist_transform.shape, dtype=np.int32)
    for i, peak in enumerate(local_maxima):
        markers[peak[0], peak[1]] = i + 1

    labels = watershed(-dist_transform, markers, mask=binary_image)
    return labels


def find_bounding_boxes(binary_image):
    """Restituisce tutte le box candidate (senza NMS)"""
    try:
        watershed_labels = watershed_segmentation(binary_image)
        use_watershed = True
    except:
        use_watershed = False
        watershed_labels = None

    boxes = []
    contours = []

    if use_watershed and watershed_labels is not None:
        unique_labels = np.unique(watershed_labels)
        for label_id in unique_labels:
            if label_id == 0:
                continue
            mask = (watershed_labels == label_id).astype(np.uint8) * 255
            label_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in label_contours:
                area = cv2.contourArea(contour)
                if area < 200:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if aspect_ratio > 10 or aspect_ratio < 0.1:
                    continue
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(binary_image.shape[1] - x, w + 2 * padding)
                h = min(binary_image.shape[0] - y, h + 2 * padding)
                boxes.append((x, y, w, h))
                contours.append(contour)
    else:
        contours_found, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_found:
            area = cv2.contourArea(contour)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(binary_image.shape[1] - x, w + 2 * padding)
            h = min(binary_image.shape[0] - y, h + 2 * padding)
            boxes.append((x, y, w, h))
            contours.append(contour)

    return boxes, contours


def score_boxes(boxes, edge_mask, depth_map):
    H, W = edge_mask.shape
    center_x, center_y = W // 2, H // 2
    scores = []

    for (x, y, w, h) in boxes:
        box_mask = np.zeros_like(edge_mask, dtype=np.uint8)
        cv2.rectangle(box_mask, (x, y), (x + w, y + h), 1, -1)

        overlap = np.logical_and(box_mask, edge_mask > 0).sum()
        overlap_ratio = overlap / (w * h + 1e-6)

        box_depth = depth_map[y:y+h, x:x+w]
        valid_depth = box_depth[box_depth > 0]
        mean_depth = valid_depth.mean() if valid_depth.size > 0 else 0

        area = w * h
        box_cx = x + w // 2
        box_cy = y + h // 2
        dist_to_center = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2)
        centrality = 1 - dist_to_center / np.sqrt(H**2 + W**2)

        score = (
            0.15 * (area / (H * W)) +
            0.35 * mean_depth +
            0.35 * overlap_ratio +
            0.15 * centrality
        )

        scores.append(score)

    return scores


def find_best_box(image_path_left, image_path_right):
    edge_mask, _, _ = edge_pipeline_py(str(image_path_left))
    depth_map = relative_depth_map(str(image_path_left), str(image_path_right))
    boxes, _ = find_bounding_boxes(edge_mask)

    if not boxes:
        return None

    scores = score_boxes(boxes, edge_mask, depth_map)
    best_idx = np.argmax(scores)
    return boxes[best_idx]
