import cv2

def find_bounding_boxes(binary_image):
    # Trova contorni esterni
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtro opzionale: ignora oggetti troppo piccoli
        if w * h > 100:  # soglia da regolare
            boxes.append((x, y, w, h))
    
    return boxes, contours
