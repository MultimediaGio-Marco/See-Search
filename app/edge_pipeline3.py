import cv2
import numpy as np

def detect_largest_object(image_path, debug=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Gradienti Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobelx, sobely)
    gradient = cv2.convertScaleAbs(gradient)

    # 2. Blur + Canny
    blurred = cv2.GaussianBlur(gradient, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 3. Morfologia per chiudere i contorni
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. Trova tutti i contorni
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nessun contorno trovato.")
        return None

    # 5. Seleziona il contorno più grande
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 6. Disegna il rettangolo sull'immagine originale
    result_img = img.copy()
    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.putText(result_img, "MAIN OBJECT", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 7. Debug visivo
    if debug:
        cv2.imshow("Gradient", gradient)
        cv2.imshow("Blurred Gradient", blurred)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("Closed", closed)
        cv2.imshow("Detected Object", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_img

# Esempio di test
detect_largest_object("../Holopix50k/val/left/-L__WfNA4a-TJ_tg28AK_left.jpg", debug=True)
detect_largest_object("./images/Mele-altasfera.jpg", debug=True)
detect_largest_object("../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg", debug=True)