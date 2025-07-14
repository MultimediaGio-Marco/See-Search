#!/usr/bin/env python3
"""
Visualizza **tutte le fasi intermedie** e le box trovate.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from find_bounding_boxes import find_bounding_boxes, pick_best_box
from edge_pipeline import edge_pipeline_full   # restituisce 4 immagini

DATA_DIR = Path("../Holopix50k/val")
LEFT  = DATA_DIR / "left"  / "-L_2ZeSXEi4XfyDTqSPg_left.jpg"
RIGHT = DATA_DIR / "right" / "-L_2ZeSXEi4XfyDTqSPg_right.jpg"

# ------------------------------------------------------------------
# pipeline
original, canny_mag, wavelet, binary = edge_pipeline_full(str(LEFT))
boxes, _ = find_bounding_boxes(binary)
best     = pick_best_box(boxes, str(LEFT), str(RIGHT))

# ------------------------------------------------------------------
# disegna box sull'originale
img_show = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
for (x, y, w, h) in boxes:
    cv2.rectangle(img_show, (x, y), (x + w, y + h), (255, 0, 0), 1)
if best is not None:
    x, y, w, h = best
    cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ------------------------------------------------------------------
# plot 5-subplot
titles = ['Originale', 'Canny+Magnitude', 'Wavelet denoise', 'Binaria finale', 'Boxes']
images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), canny_mag, wavelet, binary, img_show]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, img, ttl in zip(axes, images, titles):
    ax.imshow(img, cmap='gray' if len(img.shape)==2 else None)
    ax.set_title(ttl)
    ax.axis('off')
plt.tight_layout()
plt.show()