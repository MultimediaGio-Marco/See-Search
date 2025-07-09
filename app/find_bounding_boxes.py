import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from scipy import ndimage as ndi
from scipy.ndimage import label, distance_transform_edt

def improved_watershed_segmentation(binary_image, min_distance=15):
    """Improved watershed segmentation with better parameter tuning"""
    # Distance transform with better smoothing
    dist_transform = distance_transform_edt(binary_image)
    
    # Apply Gaussian smoothing to distance transform for better peak detection
    dist_smooth = cv2.GaussianBlur(dist_transform, (5, 5), 0)
    
    # Find local maxima with adaptive parameters
    # Use a percentage of the max distance as threshold
    threshold = 0.4 * dist_smooth.max()
    
    # Find peaks with minimum distance based on image size
    image_diagonal = np.sqrt(binary_image.shape[0]**2 + binary_image.shape[1]**2)
    adaptive_min_distance = max(10, int(image_diagonal * 0.02))
    
    local_maxima = peak_local_max(dist_smooth, 
                                   min_distance=adaptive_min_distance,
                                   threshold_abs=threshold,
                                   exclude_border=True)
    
    # Create markers for watershed
    markers = np.zeros(dist_transform.shape, dtype=np.int32)
    for i, peak in enumerate(local_maxima):
        markers[peak[0], peak[1]] = i + 1
    
    # Apply watershed algorithm
    labels = watershed(-dist_transform, markers, mask=binary_image)
    
    return labels, len(local_maxima)

def smart_contour_filtering(contour, binary_image):
    """Smart filtering of contours based on multiple criteria"""
    # Basic area filtering
    area = cv2.contourArea(contour)
    if area < 150:
        return False
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Aspect ratio filtering
    aspect_ratio = w / h
    if aspect_ratio > 8 or aspect_ratio < 0.125:  # More permissive
        return False
    
    # Extent (ratio of contour area to bounding rectangle area)
    rect_area = w * h
    extent = area / rect_area
    if extent < 0.15:  # Filter out very sparse objects
        return False
    
    # Solidity (ratio of contour area to convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
        if solidity < 0.3:  # Filter out very concave objects
            return False
    
    # Size relative to image
    image_area = binary_image.shape[0] * binary_image.shape[1]
    relative_size = area / image_area
    if relative_size > 0.8:  # Filter out objects that are too large
        return False
    
    return True

def enhanced_non_max_suppression(boxes, overlap_threshold=0.3, size_threshold=0.1):
    """Enhanced NMS with size-based filtering"""
    if len(boxes) == 0:
        return []
    
    # Convert to (x1, y1, x2, y2) format and calculate areas
    boxes_array = np.array([(x, y, x+w, y+h) for x, y, w, h in boxes])
    areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
    
    # Filter out boxes that are too small compared to the largest
    max_area = np.max(areas)
    size_mask = areas > (max_area * size_threshold)
    boxes_array = boxes_array[size_mask]
    areas = areas[size_mask]
    
    if len(boxes_array) == 0:
        return []
    
    # Sort by area (largest first)
    indices = np.argsort(areas)[::-1]
    
    keep = []
    while len(indices) > 0:
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
    """Significantly improved bounding box detection"""
    boxes = []
    contours = []
    
    # Try watershed segmentation first
    try:
        watershed_labels, num_peaks = improved_watershed_segmentation(binary_image)
        use_watershed = num_peaks > 1
        print(f"Watershed found {num_peaks} potential objects")
    except Exception as e:
        print(f"Watershed failed: {e}")
        use_watershed = False
        watershed_labels = None
    
    if use_watershed and watershed_labels is not None:
        # Process watershed labels
        unique_labels = np.unique(watershed_labels)
        
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue
                
            # Create mask for this label
            mask = (watershed_labels == label_id).astype(np.uint8) * 255
            
            # Find contours for this mask
            label_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in label_contours:
                if not smart_contour_filtering(contour, binary_image):
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # Adaptive padding based on object size
                padding = max(3, min(10, int(np.sqrt(w*h) * 0.1)))
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(binary_image.shape[1] - x, w + 2*padding)
                h = min(binary_image.shape[0] - y, h + 2*padding)
                
                boxes.append((x, y, w, h))
                contours.append(contour)
    
    # If watershed didn't work well or found too few objects, use traditional method
    if len(boxes) < 2:
        print("Using traditional contour detection as fallback")
        # Traditional contour detection with improvements
        contours_found, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []  # Reset boxes
        for contour in contours_found:
            if not smart_contour_filtering(contour, binary_image):
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Adaptive padding
            padding = max(3, min(10, int(np.sqrt(w*h) * 0.1)))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(binary_image.shape[1] - x, w + 2*padding)
            h = min(binary_image.shape[0] - y, h + 2*padding)
            
            boxes.append((x, y, w, h))
            
        contours = contours_found
    
    # Apply enhanced non-maximum suppression
    final_boxes = enhanced_non_max_suppression(boxes, overlap_threshold=0.25, size_threshold=0.1)
    
    print(f"Final detection: {len(final_boxes)} objects after filtering and NMS")
    
    return final_boxes, contours

def visualize_intermediate_results(binary_image, watershed_labels=None):
    """Helper function to visualize intermediate results for debugging"""
    if watershed_labels is not None:
        # Create colored watershed visualization
        colored_labels = np.zeros((*watershed_labels.shape, 3), dtype=np.uint8)
        for label_id in np.unique(watershed_labels):
            if label_id == 0:
                continue
            mask = watershed_labels == label_id
            colored_labels[mask] = np.random.randint(0, 255, 3)
        
        return colored_labels
    else:
        return cv2.applyColorMap(binary_image, cv2.COLORMAP_JET)