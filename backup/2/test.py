import cv2
import matplotlib.pyplot as plt
from find_bounding_boxes import find_best_box

left_img_path = "../Holopix50k/val/left/-L__uMAz3k25WltzLheY_left.jpg"
right_img_path = "../Holopix50k/val/right/-L__uMAz3k25WltzLheY_right.jpg"

# Trova la box migliore
best_box = find_best_box(left_img_path, right_img_path)
print("Best box:", best_box)

# Carica immagine RGB
image = cv2.imread(left_img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Disegna la box (x, y, w, h)
if best_box:
    x, y, w, h = best_box
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Mostra l'immagine con matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Immagine con Best Box")
plt.axis("off")
plt.tight_layout()
plt.show()



    #test_image_path = "../Holopix50k/val/left/-L__lwrICsexa0oiALCB_left.jpg"  # <-- cambia con il percorso reale
    #test_image_path = "../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg"
    
