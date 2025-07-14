import cv2
import numpy as np
import matplotlib.pyplot as plt

def improved_depth_detection(left_img_path, right_img_path):
    # Caricamento immagini
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    left_color = cv2.imread(left_img_path)
    
    if left_img is None or right_img is None:
        print("Errore nel caricamento delle immagini")
        return None
    
    # Preprocessing: equalizzazione istogramma per migliorare contrasto
    left_img = cv2.equalizeHist(left_img)
    right_img = cv2.equalizeHist(right_img)
    
    # Parametri SGBM ottimizzati per oggetti piccoli
    numDisparities = 96  # Ridotto per oggetti vicini
    blockSize = 3        # Ridotto per dettagli fini
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * 3 * blockSize**2,
        P2=32 * 3 * blockSize**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,      # Ridotto per più permissività
        speckleWindowSize=50,   # Ridotto
        speckleRange=16,        # Ridotto
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Calcolo disparità
    disparity = stereo.compute(right_img,left_img).astype(np.float32) / 16.0
    
    # Filtro per disparità valide (più permissivo)
    valid_mask = (disparity > 0) & (disparity < numDisparities-1)
    
    # Applica filtro bilaterale per smussare preservando bordi
    disparity_filtered = cv2.bilateralFilter(
        disparity.astype(np.float32), 9, 75, 75
    )
    
    # Strategia multi-soglia per catturare oggetti a diverse distanze
    results = []
    thresholds = [0.9, 0.8, 0.7, 0.6]  # Diverse soglie di vicinanza
    
    for i, threshold in enumerate(thresholds):
        if np.any(valid_mask):
            max_disp = np.max(disparity_filtered[valid_mask])
            proximity_mask = (disparity_filtered >= max_disp * threshold) & valid_mask
            
            # Pulizia morfologica adattiva
            kernel_size = max(3, 7 - i*2)  # Kernel più piccolo per soglie più basse
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            cleaned_mask = cv2.morphologyEx(
                proximity_mask.astype(np.uint8), 
                cv2.MORPH_CLOSE, 
                kernel
            )
            
            # Trova contorni
            contours, _ = cv2.findContours(
                cleaned_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Filtra contorni per area minima
                min_area = 100  # Area minima per essere considerato oggetto
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                
                if valid_contours:
                    # Ordina per area
                    valid_contours.sort(key=cv2.contourArea, reverse=True)
                    
                    result_img = left_color.copy()
                    
                    # Disegna tutti i contorni validi
                    for j, contour in enumerate(valid_contours[:3]):  # Massimo 3 oggetti
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Colori diversi per diversi oggetti
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
                        color = colors[j % len(colors)]
                        
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(result_img, f'Obj{j+1}', (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    results.append({
                        'threshold': threshold,
                        'image': result_img,
                        'disparity': disparity_filtered,
                        'mask': cleaned_mask,
                        'contours': valid_contours
                    })
    
    return results

def visualize_results(results):
    """Visualizza i risultati con diverse soglie"""
    if not results:
        print("Nessun risultato da visualizzare")
        return
    
    fig, axes = plt.subplots(2, len(results), figsize=(15, 8))
    if len(results) == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results):
        # Immagine con bounding box
        axes[0, i].imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f'Soglia: {result["threshold"]:.1f}')
        axes[0, i].axis('off')
        
        # Mappa disparità
        axes[1, i].imshow(result['disparity'], cmap='jet')
        axes[1, i].set_title('Mappa Disparità')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def detect_bird_specifically(left_img_path, right_img_path):
    """Funzione specifica per rilevare uccelli/piccioni"""
    
    # Usa la funzione migliorata
    results = improved_depth_detection(left_img_path, right_img_path)
    
    if results:
        print(f"Rilevati {len(results)} risultati con diverse soglie")
        
        # Analizza ogni risultato
        for i, result in enumerate(results):
            print(f"\nSoglia {result['threshold']:.1f}:")
            print(f"  - Oggetti rilevati: {len(result['contours'])}")
            
            for j, contour in enumerate(result['contours'][:3]):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                print(f"  - Oggetto {j+1}: Area={area:.0f}, Pos=({x},{y}), Dim={w}x{h}")
        
        # Visualizza risultati
        visualize_results(results)
        
        return results
    else:
        print("Nessun oggetto rilevato")
        return None

# Esempio d'uso
results = detect_bird_specifically(
    "../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg",
    "../Holopix50k/val/right/-LahPsJhCZTWwgvaAMB4_right.jpg"
)

# Versione alternativa usando anche la segmentazione colore
def detect_bird_with_color_segmentation(left_img_path, right_img_path):
    """Combina stereo vision con segmentazione colore"""
    
    left_color = cv2.imread(left_img_path)
    if left_color is None:
        return None
    
    # Segmentazione colore per uccelli grigi/bianchi
    hsv = cv2.cvtColor(left_color, cv2.COLOR_BGR2HSV)
    
    # Range per colori del piccione (grigio/bianco)
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 30, 255])
    
    color_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Pulizia morfologica
    kernel = np.ones((5,5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # Trova contorni nella maschera colore
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filtra per area e forma
        bird_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # Area ragionevole per un piccione
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Rapporto aspetto ragionevole
                    bird_contours.append(contour)
        
        # Disegna risultati
        result_img = left_color.copy()
        for contour in bird_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(result_img, 'Bird', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return result_img, color_mask
    
    return None, None

# Testa anche la segmentazione colore
bird_result, color_mask = detect_bird_with_color_segmentation(
    "../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg",
    "../Holopix50k/val/right/-LahPsJhCZTWwgvaAMB4_right.jpg"
)

if bird_result is not None:
    cv2.imshow("Bird Detection - Color", bird_result)
    cv2.imshow("Color Mask", color_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()