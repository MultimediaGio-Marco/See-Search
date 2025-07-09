import argparse
import cv2
import numpy as np
from PIL import Image
import os

from edge_pipeline import edge_pipeline_py
from find_bounding_boxes import find_bounding_boxes

def draw_boxes(image, boxes, color=(0,255,0), thickness=2):
    """Draw bounding boxes on image"""
    for (x,y,w,h) in boxes:
        cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness)
    return image

def main():
    parser = argparse.ArgumentParser(description="Improved Edge pipeline + bounding boxes")
    parser.add_argument("input",  help="Path to input image")
    parser.add_argument("--out-mask",   default="mask.png",   help="Path to save binary mask")
    parser.add_argument("--out-mag",    default="mag.png",    help="Path to save magnitude")
    parser.add_argument("--out-den",    default="den.png",    help="Path to save denoised")
    parser.add_argument("--out-boxes",  default="result.png",  help="Path to save boxed overlay")
    parser.add_argument("--min-area",   type=int, default=200,  help="Min area for filtering boxes")
    args = parser.parse_args()

    try:
        # [1] Esegui pipeline migliorata
        print("Running improved edge detection pipeline...")
        bw, mag, den = edge_pipeline_py(args.input)
        print(f"Edge detection completed. Found {np.sum(bw > 0)} edge pixels.")

        # [2] Crea cartella di output
        output_path = "./images/Result/"+ os.path.basename(args.input).split('.')[0]
        print(f"Saving results to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        # [3] Percorsi file di output
        mask_path  = os.path.join(output_path, args.out_mask)
        mag_path   = os.path.join(output_path, args.out_mag)
        den_path   = os.path.join(output_path, args.out_den)
        boxes_path = os.path.join(output_path, args.out_boxes)

        # [4] Salva immagini intermedie
        Image.fromarray(bw).save(mask_path)
        Image.fromarray(mag).save(mag_path)
        Image.fromarray(den).save(den_path)
        print("Intermediate images saved.")

        # [5] Trova bounding boxes con algoritmo migliorato
        print("Finding bounding boxes with improved algorithm...")
        boxes, contours = find_bounding_boxes(bw)
        print(f"Found {len(boxes)} objects after filtering and NMS.")

        # [6] Disegna bounding boxes su immagine originale
        orig = cv2.imread(args.input)
        if orig is None:
            raise ValueError(f"Could not load image: {args.input}")
        
        boxed = draw_boxes(orig.copy(), boxes)

        # [7] Salva risultato finale
        cv2.imwrite(boxes_path, boxed)
        
        # [8] Stampa riepilogo
        print("\n" + "="*50)
        print("RESULTS SUMMARY:")
        print("="*50)
        print(f"Input image: {args.input}")
        print(f"Objects detected: {len(boxes)}")
        print(f"Output directory: {output_path}")
        print("\nFiles saved:")
        print(f" - Binary mask: {args.out_mask}")
        print(f" - Edge magnitude: {args.out_mag}")
        print(f" - Denoised edges: {args.out_den}")
        print(f" - Final result: {args.out_boxes}")
        
        if len(boxes) > 0:
            print("\nBounding boxes (x, y, width, height):")
            for i, (x, y, w, h) in enumerate(boxes):
                print(f"  Object {i+1}: ({x}, {y}, {w}, {h}) - Area: {w*h}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()