#from disparita import user_disparity
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import pathlib as path

# Crea matcher stereo

def relative_depth_map(left_img_path, right_img_path):
    # Carica le immagini in scala di grigi
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Calcola la mappa di disparità
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # deve essere multiplo di 16
        blockSize=5,
        P1=8 * 1 * 5**2,
        P2=32 * 1 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    
    # Maschera valida: solo disparità positive
    valid_mask = disparity > 0
    safe_disparity = np.where(valid_mask, disparity, 1.0)  # evita divisione per zero o negativi

    # Calcolo profondità relativa (solo su disparità valide)
    relative_depth = np.zeros_like(disparity)
    relative_depth[valid_mask] = np.log1p(1 / (safe_disparity[valid_mask]))

    # Normalizzazione su valori validi
    normalized_depth = np.zeros_like(relative_depth)
    valid_depths = relative_depth[valid_mask]
    if valid_depths.size > 0:
        normalized_depth[valid_mask] = (
            (valid_depths - valid_depths.min()) /
            (valid_depths.max() - valid_depths.min() + 1e-6)
        )

    return normalized_depth

    
def display_relative_depth_map(left_img_path, right_img_path):
    normalized_depth= relative_depth_map(left_img_path, right_img_path)
    # Visualizzazione
    plt.figure(figsize=(10, 6))
    img = plt.imshow(normalized_depth, cmap='inferno')
    plt.title('Relative Depth Map (normalized)')
    plt.axis('off')
    cbar = plt.colorbar(img, shrink=0.7)
    cbar.set_label('Normalized Relative Depth')
    plt.tight_layout()
    plt.show()

#display_relative_depth_map('../Holopix50k/val/left/-L__uMAz3k25WltzLheY_left.jpg', '../Holopix50k/val/right/-L__uMAz3k25WltzLheY_right.jpg')
def getImg():
    image_Path = '../Holopix50k/val'
    image_Path = path.Path(image_Path)
    Ll=sorted(list(image_Path.glob('left/*.jpg')))
    Lr=sorted(list(image_Path.glob('right/*.jpg')))
    random_image = random.randint(0, len(Ll) - 1)
    print(f"Selected image index: {random_image}")
    left_img_path = Ll[random_image]
    print(f"Left image path: {left_img_path}")
    right_img_path = Lr[random_image]
    
    # Calcola la mappa di disparità
    display_relative_depth_map(left_img_path, right_img_path)
    

    
#getImg()