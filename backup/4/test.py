#!/usr/bin/env python3
"""
test_pipeline.py – test rapido per edge_pipeline + find_boxes
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from edge_pipeline import edge_pipeline_py
from find_bounding_boxes import find_bounding_boxes

def run_full_test(image_path,
                  flex=None,
                  show=False,
                  save_results=True,
                  out_dir="./test_outputs"):
    flex = flex or {}

    image_rgb = cv2.imread(image_path)
    if image_rgb is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    bw_final, combined_edges, denoised_norm = edge_pipeline_py(image_path, flex=flex)

    largest_box, contours = find_bounding_boxes(
        bw_final,
        min_area=flex.get('min_area', 200),
        max_aspect=flex.get('max_aspect', 10),
        nms_iou=flex.get('nms_iou', 0.3),
        padding=flex.get('padding', 5)
    )
    boxes = [largest_box] if largest_box else []

    image_with_boxes = image_rgb.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show:
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        fig.suptitle("Edge Detection + Bounding Boxes", fontsize=16)

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(combined_edges, cmap="gray")
        axes[1].set_title("Combined Edges")
        axes[1].axis("off")

        axes[2].imshow(bw_final, cmap="gray")
        axes[2].set_title("Binary Mask")
        axes[2].axis("off")

        axes[3].imshow(image_with_boxes)
        axes[3].set_title(f"Detected Boxes ({len(boxes)})")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()

    if save_results:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(os.path.join(out_dir, f"{base}_binary_mask.png"), bw_final)
        cv2.imwrite(os.path.join(out_dir, f"{base}_edges.png"), combined_edges)
        cv2.imwrite(os.path.join(out_dir, f"{base}_with_boxes.png"),
                    cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        print(f"[✔] Risultati salvati in {out_dir}")

# -----------------------------
# ESEMPIO DI CHIAMATA SEMPLICE
# -----------------------------
run_full_test(
    image_path="./images/Mele-altasfera.jpg",  # <- cambia con il tuo path
    flex={
        'min_area': 100,
        'max_aspect': 8,
        'nms_iou': 0.4,
        'padding': 5,
        'a_weight': 1.2,
        'b_weight': 1.0,
        'close_ks': 5,
        'open_ks': 3,
        'fill_holes': True
    },
    show=True
)
