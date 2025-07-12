import cv2
import numpy as np

# === Caricamento immagine ===
img_path = "../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg"
img_path = "./images/Mele-altasfera.jpg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Immagine non trovata")

# Resize (coerente con l'originale)
img = cv2.resize(img, (640, 480))
h, w = img.shape[:2]

# === Definizione del rettangolo di delimitazione ===
x = int(w * 0.1)
y = int(h * 0.1)
width = int(w * 0.8)
height = int(h * 0.8)

# === Approccio con filtri tradizionali ===
# 1. Estrai la ROI (Region of Interest)
roi = img[y:y+height, x:x+width]

# 2. Converti in spazio colore HSV per migliore segmentazione
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 3. Calcola istogramma dei colori nella ROI
h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])

# 4. Trova il range di colori dominante
h_dominant = np.argmax(h_hist)
s_dominant = np.argmax(s_hist)

# 5. Definisci range di colori per la maschera
lower_bound = np.array([max(0, h_dominant - 20), max(0, s_dominant - 40), 50])
upper_bound = np.array([min(180, h_dominant + 20), min(255, s_dominant + 40), 255])

# 6. Applica thresholding all'intera immagine
full_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(full_hsv, lower_bound, upper_bound)

# 7. Operazioni morfologiche per affinare la maschera
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 8. Trova il contorno principale
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    main_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
else:
    final_mask = mask

# 9. Applica la maschera finale
segmented = cv2.bitwise_and(img, img, mask=final_mask)

# === Visualizzazione risultati ===
img_with_rect = img.copy()
cv2.rectangle(img_with_rect, (x, y), (x + width, y + height), (0, 255, 0), 2)

cv2.imshow("Immagine Originale con Rettangolo", img_with_rect)
cv2.imshow("Segmentazione Tradizionale", segmented)
cv2.imshow("Maschera Finale", final_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()