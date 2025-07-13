import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def watershed_segmentation(binary_image, min_distance=20):
    """Use watershed to separate touching objects"""
    # Distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    # Find local maxima (peaks)
    local_maxima = peak_local_max(dist_transform, min_distance=min_distance, 
                                threshold_abs=0.3*dist_transform.max())
    
    # Create markers
    markers = np.zeros(dist_transform.shape, dtype=np.int32)
    for i, peak in enumerate(local_maxima):
        markers[peak[0], peak[1]] = i + 1
    
    # Apply watershed
    labels = watershed(-dist_transform, markers, mask=binary_image)
    
    return labels

def non_max_suppression(boxes, overlap_threshold=0.3):
    """Remove overlapping bounding boxes"""
    if len(boxes) == 0:
        return []
    
    # Convert to (x1, y1, x2, y2) format
    boxes_array = np.array([(x, y, x+w, y+h) for x, y, w, h in boxes])
    
    # Calculate areas
    areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
    
    # Sort by area (largest first)
    indices = np.argsort(areas)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep the largest box
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes_array[current]
        remaining_boxes = boxes_array[indices[1:]]
        
        # Intersection coordinates
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        # Intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Union area
        union = areas[current] + areas[indices[1:]] - intersection
        
        # IoU
        iou = intersection / union
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][iou < overlap_threshold]
    
    # Convert back to (x, y, w, h) format
    final_boxes = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_array[idx]
        final_boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
    
    return final_boxes

def find_bounding_boxes(binary_image):
    """Return only the largest bounding box using watershed + NMS"""
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

    # Applica NMS
    final_boxes = non_max_suppression(boxes, overlap_threshold=0.3)

    if not final_boxes:
        return None, []

    # Trova la box con area massima
    largest_box = max(final_boxes, key=lambda b: b[2] * b[3])
    return largest_box, contours
