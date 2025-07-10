import cv2
import numpy as np


def calculate_disparity(left, right, block_size=5, max_disparity=64):
    """Calculate disparity between two images using block matching."""
    h, w = left.shape[:2]
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(max_disparity, w - block_size + 1, block_size):
            block1 = left[y:y + block_size, x:x + block_size]
            min_diff = float('inf')
            best_disparity = 0

            for dx in range(max_disparity):
                x_right = x - dx
                if x_right < 0 or x_right + block_size > w:
                    continue
                block2 = right[y:y + block_size, x_right:x_right + block_size]
                diff = np.sum(np.abs(block1 - block2))
                if diff < min_diff:
                    min_diff = diff
                    best_disparity = dx

            disparity_map[y:y + block_size, x:x + block_size] = best_disparity

    return disparity_map

def user_disparity(left_image, right_image, block_size=5, max_disparity=64):
    """
    User-facing function to calculate disparity between two images.
    
    Parameters:
    - left_image: Path to the left image.
    - right_image: Path to the right image.
    - block_size: Size of the blocks for matching.
    - max_disparity: Maximum disparity to consider.
    
    Returns:
    - Disparity map as a numpy array.
    """
    left = cv2.imread(left_image, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_image, cv2.IMREAD_GRAYSCALE)

    if left is None or right is None:
        raise ValueError("Could not read one of the images. Check paths.")

    return calculate_disparity(left, right, block_size, max_disparity)
