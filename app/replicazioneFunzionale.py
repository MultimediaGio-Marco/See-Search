import cv2
import numpy as np

def traditional_bounding_box(img):
    """
    Segmentazione robusta con filtri tradizionali per il bounding box
    Restituisce maschera binaria e rettangolo (x, y, w, h)
    """
    h, w = img.shape[:2]
    
    # 1. Riduzione rumore e miglioramento contrasto
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 2. Calcolo automatico dei valori di soglia
    v_channel = hsv[:, :, 2]
    _, thresh = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Operazioni morfologiche per pulire la maschera
    kernel = np.ones((7, 7), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Rilevamento contorni e selezione principale
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: bounding box centrale
        x, y, w_box, h_box = int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6)
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_mask[y:y+h_box, x:x+w_box] = 255
        return final_mask, (x, y, w_box, h_box)
    
    # Trova il contorno più grande
    main_contour = max(contours, key=cv2.contourArea)
    
    # 5. Creazione maschera dal contorno
    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
    
    # 6. Calcolo bounding box
    x, y, w_box, h_box = cv2.boundingRect(main_contour)
    
    return final_mask, (x, y, w_box, h_box)

def main():
    # === Caricamento e ridimensionamento ===
    img_path = "../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg"
    img_path ="../Holopix50k/val/left/-L__lwrICsexa0oiALCB_left.jpg"
    img = cv2.imread(img_path)
    if img is None:
        # Prova un percorso alternativo
        img_path = "./images/example.jpg"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError("Immagine non trovata")

    img = cv2.resize(img, (640, 480))
    original = img.copy()

    # === Segmentazione avanzata ===
    final_mask, (x, y, w_box, h_box) = traditional_bounding_box(img)

    # === Applicazione maschera ===
    segmented = cv2.bitwise_and(img, img, mask=final_mask)

    # === Disegna il bounding box ===
    cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    # === Visualizzazione ===
    cv2.imshow("Originale con Bounding Box", img)
    cv2.imshow("Segmentazione", segmented)
    cv2.imshow("Maschera Finale", final_mask)
    
    print(f"Bounding box: x={x}, y={y}, width={w_box}, height={h_box}")
    print(f"Area oggetto: {w_box * h_box} px")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()