import argparse
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Import your improved modules
from edge_pipeline import edge_pipeline_py
from find_bounding_boxes import find_bounding_boxes

def draw_boxes_with_labels(image, boxes, color=(0,255,0), thickness=2):
    """Draw bounding boxes with labels on image"""
    result = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        
        # Add label
        label = f"Obj {i+1}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result, (x, y-label_size[1]-5), (x+label_size[0], y), color, -1)
        cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Add area info
        area_text = f"Area: {w*h}"
        cv2.putText(result, area_text, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return result

def create_comparison_image(original, binary_mask, final_result):
    """Create a side-by-side comparison image"""
    # Resize images to same height
    height = min(original.shape[0], binary_mask.shape[0], final_result.shape[0])
    
    # Resize original
    scale = height / original.shape[0]
    new_width = int(original.shape[1] * scale)
    original_resized = cv2.resize(original, (new_width, height))
    
    # Convert binary mask to 3-channel
    binary_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    binary_resized = cv2.resize(binary_3ch, (new_width, height))
    
    # Resize final result
    final_resized = cv2.resize(final_result, (new_width, height))
    
    # Concatenate horizontally
    comparison = np.hstack([original_resized, binary_resized, final_resized])
    
    return comparison

def analyze_objects(boxes, image_shape):
    """Analyze detected objects and provide statistics"""
    if not boxes:
        return "No objects detected"
    
    areas = [w*h for x, y, w, h in boxes]
    aspect_ratios = [w/h for x, y, w, h in boxes]
    
    stats = f"""
OBJECT ANALYSIS:
================
Total objects: {len(boxes)}
Area statistics:
  - Min area: {min(areas)}
  - Max area: {max(areas)}
  - Mean area: {np.mean(areas):.1f}
  - Median area: {np.median(areas):.1f}

Aspect ratio statistics:
  - Min ratio: {min(aspect_ratios):.2f}
  - Max ratio: {max(aspect_ratios):.2f}
  - Mean ratio: {np.mean(aspect_ratios):.2f}

Image coverage: {sum(areas) / (image_shape[0] * image_shape[1]) * 100:.1f}%
"""
    return stats

def main():
    parser = argparse.ArgumentParser(description="Significantly Improved Object Detection Pipeline")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--out-mask", default="mask.png", help="Binary mask output")
    parser.add_argument("--out-mag", default="mag.png", help="Edge magnitude output")
    parser.add_argument("--out-den", default="den.png", help="Denoised output")
    parser.add_argument("--out-boxes", default="result.png", help="Final result with boxes")
    parser.add_argument("--out-comparison", default="comparison.png", help="Side-by-side comparison")
    parser.add_argument("--out-analysis", default="analysis.txt", help="Object analysis report")
    parser.add_argument("--debug", action="store_true", help="Save debug visualizations")
    args = parser.parse_args()

    try:
        # Validate input
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input image not found: {args.input}")
        
        print("="*60)
        print("IMPROVED OBJECT DETECTION PIPELINE")
        print("="*60)
        print(f"Processing: {args.input}")
        
        # [1] Run improved edge detection pipeline
        print("\n[1/5] Running enhanced edge detection...")
        bw, mag, den = edge_pipeline_py(args.input)
        print(f"    Edge detection completed. Found {np.sum(bw > 0)} edge pixels.")

        # [2] Create output directory
        output_dir = os.path.join("./images/Result", os.path.basename(args.input).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)
        print(f"    Output directory: {output_dir}")
        
        # [3] Save intermediate results
        print("\n[2/5] Saving intermediate results...")
        Image.fromarray(bw).save(os.path.join(output_dir, args.out_mask))
        Image.fromarray(mag).save(os.path.join(output_dir, args.out_mag))
        Image.fromarray(den).save(os.path.join(output_dir, args.out_den))
        
        # [4] Find bounding boxes with improved algorithm
        print("\n[3/5] Detecting objects with advanced algorithm...")
        boxes, contours = find_bounding_boxes(bw)
        
        # [5] Load original image and create visualizations
        print("\n[4/5] Creating visualizations...")
        original = cv2.imread(args.input)
        if original is None:
            raise ValueError(f"Could not load image: {args.input}")
        
        # Draw boxes with labels
        final_result = draw_boxes_with_labels(original, boxes, color=(0, 255, 0), thickness=3)
        
        # Save final result
        cv2.imwrite(os.path.join(output_dir, args.out_boxes), final_result)
        
        # Create comparison image
        comparison = create_comparison_image(original, bw, final_result)
        cv2.imwrite(os.path.join(output_dir, args.out_comparison), comparison)
        
        # [6] Generate analysis report
        print("\n[5/5] Generating analysis report...")
        analysis = analyze_objects(boxes, original.shape)
        
        # Save analysis to file
        with open(os.path.join(output_dir, args.out_analysis), 'w') as f:
            f.write(f"Object Detection Analysis Report\n")
            f.write(f"Input: {args.input}\n")
            f.write(f"Generated: {os.path.basename(__file__)}\n")
            f.write("="*50 + "\n")
            f.write(analysis)
            f.write("\n\nDetailed Object Information:\n")
            f.write("-" * 30 + "\n")
            for i, (x, y, w, h) in enumerate(boxes):
                f.write(f"Object {i+1}: Position({x}, {y}), Size({w}x{h}), Area={w*h}, Aspect={w/h:.2f}\n")
        
        # [7] Debug visualizations
        if args.debug:
            print("\n[DEBUG] Saving debug visualizations...")
            # You can add watershed visualization here if needed
            # debug_vis = visualize_intermediate_results(bw, watershed_labels)
            # cv2.imwrite(os.path.join(output_dir, "debug_watershed.png"), debug_vis)
        
        # [8] Final summary
        print("\n" + "="*60)
        print("DETECTION RESULTS:")
        print("="*60)
        print(f"✓ Objects detected: {len(boxes)}")
        print(f"✓ Output saved to: {output_dir}")
        print(f"✓ Files created:")
        print(f"    - {args.out_mask} (binary mask)")
        print(f"    - {args.out_mag} (edge magnitude)")
        print(f"    - {args.out_den} (denoised edges)")
        print(f"    - {args.out_boxes} (final result)")
        print(f"    - {args.out_comparison} (comparison view)")
        print(f"    - {args.out_analysis} (analysis report)")
        
        print(analysis)
        
        if len(boxes) > 0:
            print("\nDetected Objects (x, y, width, height):")
            print("-" * 40)
            for i, (x, y, w, h) in enumerate(boxes):
                print(f"  Object {i+1:2d}: ({x:3d}, {y:3d}, {w:3d}, {h:3d}) | Area: {w*h:5d} | Ratio: {w/h:.2f}")
        
        print("="*60)
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())