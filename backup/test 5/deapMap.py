"""
Stereo-based relative depth map (pure OpenCV).
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Stereo matcher parameters
# ------------------------------------------------------------------
STEREO_PARAMS = dict(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8 * 1 * 5 ** 2,
    P2=32 * 1 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
def relative_depth_map(left_path: str | Path, right_path: str | Path) -> np.ndarray:
    left  = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    stereo = cv2.StereoSGBM_create(**STEREO_PARAMS)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0

    valid = disp > 0
    rel = np.zeros_like(disp)
    rel[valid] = np.log1p(1 / disp[valid])

    # normalize 0-1
    v = rel[valid]
    if v.size:
        rel[valid] = (v - v.min()) / (v.max() - v.min() + 1e-6)

    return rel


# ------------------------------------------------------------------
# Quick visualisation helper
# ------------------------------------------------------------------
def display_relative_depth_map(left: str | Path, right: str | Path) -> None:
    depth = relative_depth_map(left, right)
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap="inferno")
    plt.title("Relative depth map")
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()