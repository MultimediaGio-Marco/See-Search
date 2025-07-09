import argparse
import cv2
import numpy as np
from PIL import Image
import os

from edge_pipeline import edge_pipeline_py  # sostituisci with il nome del file
from find_bounding_boxes import find_bounding_boxes  # sostituisci con il nome del file

def draw_boxes(image, boxes, color=(0,255,0), thickness=2):
    for (x,y,w,h) in boxes:
        cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness)
    return image

def main():
    parser = argparse.ArgumentParser(description="Edge pipeline + bounding boxes")
    parser.add_argument("input",  help="Path to input image")
    parser.add_argument("--out-mask",   default="mask.png",   help="Path to save binary mask")
    parser.add_argument("--out-mag",    default="mag.png",    help="Path to save magnitude")
    parser.add_argument("--out-den",    default="den.png",    help="Path to save denoised")
    parser.add_argument("--out-boxes",  default="boxed.png",  help="Path to save boxed overlay")
    parser.add_argument("--min-area",   type=int, default=100,  help="Min area for filtering boxes")
    args = parser.parse_args()

    # [1] Esegui pipeline
    bw, mag, den = edge_pipeline_py(args.input)

    # salva mask/mag/den
    output_path = "./images/Result/"+ os.path.basename(args.input).split('.')[0]
    print(f"Saving results to {output_path}")
    folder=os.makedirs(output_path, exist_ok=True)
    
    mask_path  = os.path.join(output_path, args.out_mask)
    mag_path   = os.path.join(output_path, args.out_mag)
    den_path   = os.path.join(output_path, args.out_den)
    boxes_path = os.path.join(output_path, args.out_boxes)

    
    Image.fromarray(bw).save(mask_path)
    Image.fromarray(mag).save(mag_path)
    Image.fromarray(den).save(den_path)

    # [2] Trova bounding boxes
    boxes, contours = find_bounding_boxes(bw)

    # [3] Disegna su immagine originale
    orig = cv2.imread(args.input)
    boxed = draw_boxes(orig.copy(), boxes)

    cv2.imwrite(boxes_path, boxed)
    print(f"Saved:\n - mask:   {args.out_mask}\n - mag:    {args.out_mag}\n - den:    {args.out_den}\n - boxed:  {args.out_boxes}")

if __name__ == "__main__":
    main()
