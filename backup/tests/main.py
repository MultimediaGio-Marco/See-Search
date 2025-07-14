import cv2
import numpy as np
import os
import random
import pathlib as path
from PIL import Image

# Import modules
from deapMap import relative_depth_map
from edge_pipeline import detect_and_segment_objects, generic_object_pipeline
from find_bounding_boxes import find_bounding_boxes


def draw_boxes_with_labels(image, boxes, color=(0,255,0), thickness=2):
    """Disegna bounding boxes con etichette informative"""
    result = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        label = f"Obj {i+1}"
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result, (x, y-size[1]-5), (x+size[0], y), color, -1)
        cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        area_text = f"Area: {w*h}"
        cv2.putText(result, area_text, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return result


def create_comparison_image(orig, depth_map, seg_mask, final):
    """Crea immagine comparativa con tutti i passaggi del pipeline"""
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
    
    # Ridimensiona tutte le immagini alla stessa altezza
    h = min(orig.shape[0], depth_color.shape[0], mask_color.shape[0], final.shape[0])
    
    def resize(img):
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h))
    
    imgs = [resize(orig), resize(depth_color), resize(mask_color), resize(final)]
    return np.hstack(imgs)


def analyze_objects(boxes, img_shape):
    """Analizza statistiche degli oggetti rilevati"""
    if not boxes:
        return "No objects detected"
    
    areas = [w*h for _,_,w,h in boxes]
    ratios = [w/h for _,_,w,h in boxes]
    
    stats = (
        f"OBJECT ANALYSIS:\n"
        f"Total objects: {len(boxes)}\n"
        f"Area - min: {min(areas)}, max: {max(areas)}, mean: {np.mean(areas):.1f}\n"
        f"Aspect ratio - min: {min(ratios):.2f}, max: {max(ratios):.2f}, mean: {np.mean(ratios):.2f}\n"
        f"Coverage: {sum(areas)/(img_shape[0]*img_shape[1])*100:.1f}%"
    )
    return stats


def select_random_pair(dataset_dir="../Holopix50k/val"):
    """Seleziona una coppia stereo casuale dal dataset"""
    image_path = path.Path(dataset_dir)
    left_list = sorted(image_path.glob('left/*.jpg'))
    right_list = sorted(image_path.glob('right/*.jpg'))
    idx = random.randint(0, len(left_list) - 1)
    return str(left_list[idx]), str(right_list[idx])


def extract_closest_objects_mask(depth_map, tolerance_factor=0.02):
    """Estrae maschera degli oggetti più vicini dalla depth map"""
    valid = depth_map > 0
    if np.any(valid):
        min_d = depth_map[valid].min()
        max_d = depth_map[valid].max()
        tol = tolerance_factor * (max_d - min_d)
        closest = ((depth_map >= min_d + tol) * 255).astype(np.uint8)
    else:
        closest = np.zeros_like(depth_map, dtype=np.uint8)
    return closest


def fuse_depth_and_segmentation(depth_mask, seg_mask, method='and'):
    """Fonde la maschera di profondità con la segmentazione"""
    if method == 'and':
        # Intersezione - solo dove entrambe le maschere sono attive
        fused = cv2.bitwise_and(depth_mask, seg_mask)
    elif method == 'or':
        # Unione - dove almeno una delle maschere è attiva
        fused = cv2.bitwise_or(depth_mask, seg_mask)
    elif method == 'weighted':
        # Combinazione pesata
        fused = cv2.addWeighted(depth_mask, 0.6, seg_mask, 0.4, 0)
        _, fused = cv2.threshold(fused, 127, 255, cv2.THRESH_BINARY)
    else:
        fused = seg_mask
    
    return fused


def process_single_image(image_path, segmentation_method='combined', 
                        apply_watershed=True, min_object_size=1000):
    """Processa una singola immagine senza informazioni di profondità"""
    print(f"[1/3] Processing image: {image_path}")
    
    # Segmentazione oggetti
    seg_mask, contours = detect_and_segment_objects(
        image_path, 
        method=segmentation_method,
        apply_watershed=apply_watershed
    )
    
    if seg_mask is None:
        print("No objects detected in the image")
        return None, None, None
    
    print(f"[2/3] Detecting bounding boxes...")
    boxes, _ = find_bounding_boxes(seg_mask)
    
    # Visualizzazione
    orig = cv2.imread(image_path)
    final = draw_boxes_with_labels(orig, boxes)
    
    return orig, seg_mask, final, boxes


def main(use_stereo=True, single_image_path=None, segmentation_method='combined'):
    """Pipeline principale - supporta sia stereo che singola immagine"""
    
    if use_stereo and single_image_path is None:
        # Modalità stereo con depth map
        print("=== STEREO MODE ===")
        left, right = select_random_pair()
        print(f"Using stereo pair: {left} and {right}")
        
        # Preparazione directory output
        base = os.path.splitext(os.path.basename(left))[0]
        out_dir = os.path.join("./images/Result", base)
        os.makedirs(out_dir, exist_ok=True)
        
        # 1) Calcola depth map
        print("[1/5] Computing depth map...")
        depth = relative_depth_map(left, right)
        depth_vis = (depth * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_vis)
        
        # 2) Estrai maschera oggetti più vicini
        print("[2/5] Extracting closest objects mask...")
        closest_mask = extract_closest_objects_mask(depth)
        
        # 3) Segmentazione generica oggetti
        print("[3/5] Running generic object segmentation...")
        seg_mask, contours = detect_and_segment_objects(
            left, 
            method=segmentation_method,
            apply_watershed=True
        )
        
        if seg_mask is None:
            print("No objects detected - using depth-only segmentation")
            seg_mask = closest_mask
        
        # 4) Fusione maschere
        print("[4/5] Fusing depth and segmentation masks...")
        fused = fuse_depth_and_segmentation(closest_mask, seg_mask, method='and')
        cv2.imwrite(os.path.join(out_dir, "mask.png"), fused)
        
        # 5) Rilevamento bounding boxes e visualizzazione
        print("[5/5] Detecting bounding boxes and saving results...")
        boxes, _ = find_bounding_boxes(fused)
        orig = cv2.imread(left)
        final = draw_boxes_with_labels(orig, boxes)
        cv2.imwrite(os.path.join(out_dir, "result.png"), final)
        
        # Crea comparazione e report
        comp = create_comparison_image(orig, depth, fused, final)
        cv2.imwrite(os.path.join(out_dir, "comparison.png"), comp)
        
        image_path = left
        
    else:
        # Modalità singola immagine
        print("=== SINGLE IMAGE MODE ===")
        if single_image_path is None:
            print("Error: single_image_path must be provided for single image mode")
            return
        
        # Preparazione directory output
        base = os.path.splitext(os.path.basename(single_image_path))[0]
        out_dir = os.path.join("./images/Result", base)
        os.makedirs(out_dir, exist_ok=True)
        
        # Processa immagine singola
        result = process_single_image(single_image_path, segmentation_method)
        if result[0] is None:
            return
            
        orig, seg_mask, final, boxes = result
        
        # Salva risultati
        cv2.imwrite(os.path.join(out_dir, "mask.png"), seg_mask)
        cv2.imwrite(os.path.join(out_dir, "result.png"), final)
        
        # Comparazione semplificata (senza depth map)
        mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
        comp = np.hstack([orig, mask_color, final])
        cv2.imwrite(os.path.join(out_dir, "comparison.png"), comp)
        
        image_path = single_image_path
        fused = seg_mask
    
    # Analisi finale
    report = analyze_objects(boxes, orig.shape)
    with open(os.path.join(out_dir, "analysis.txt"), 'w') as f:
        f.write(f"Generic Object Detection Report\n")
        f.write(f"Input: {image_path}\n")
        f.write(f"Segmentation method: {segmentation_method}\n")
        f.write(f"Use stereo: {use_stereo}\n\n")
        f.write(report)
    
    print("Processing completed. Results in:", out_dir)
    print(report)
    
    return out_dir, boxes, fused


def batch_process_images(image_dir, segmentation_method='combined'):
    """Processa un batch di immagini"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(path.Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(path.Path(image_dir).glob(f'*{ext.upper()}'))
    
    results = []
    for img_path in image_paths:
        print(f"\n--- Processing: {img_path} ---")
        try:
            out_dir, boxes, mask = main(use_stereo=False, 
                                       single_image_path=str(img_path), 
                                       segmentation_method=segmentation_method)
            results.append((str(img_path), len(boxes), out_dir))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append((str(img_path), 0, None))
    
    # Report finale
    print("\n=== BATCH PROCESSING RESULTS ===")
    for img_path, num_objects, out_dir in results:
        print(f"{os.path.basename(img_path)}: {num_objects} objects detected")
    
    return results


if __name__ == "__main__":
    # Esempi di utilizzo
    
    # 1. Modalità stereo (default)
    # main(use_stereo=True, segmentation_method='combined')
    
    # 2. Modalità singola immagine
    # main(use_stereo=False, single_image_path="path/to/your/image.jpg", segmentation_method='combined')
    
    # 3. Test diversi metodi di segmentazione
    methods = ['edges', 'gradient', 'color', 'combined']
    for method in methods:
        print(f"\n========== TESTING METHOD: {method} ==========")
        # main(use_stereo=False, single_image_path="test_image.jpg", segmentation_method=method)
    
    # 4. Batch processing
    # batch_process_images("path/to/image/directory", segmentation_method='combined')
    
    # Default: stereo mode
    main(use_stereo=True, segmentation_method='combined')