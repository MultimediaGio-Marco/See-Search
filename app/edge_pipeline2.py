import cv2
import numpy as np
import os

def detect_main_object_with_canny_tuning(image_path):
    if not os.path.exists(image_path):
        print("File non trovato:", image_path)
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parametri selezionabili
    blur_k = 15
    low_th, high_th = 30, 60
    min_area = 1000
    sigma=3

    # 1) Blur
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), sigma)
    cv2.imshow("1 - Gaussian Blur", blurred)
    cv2.waitKey(0)

    # 2) Canny
    edges = cv2.Canny(blurred, low_th, high_th)
    cv2.imshow("2 - Canny", edges)
    cv2.waitKey(0)

    # 3) Morfologia (OPEN per rimuovere contorni piccoli isolati)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("3 - Closing", closed)
    cv2.waitKey(0)

    # 4) Trova contorni
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contorni trovati: {len(contours)}")

    if not contours:
        print("❌ Nessun contorno rilevato.")
        cv2.destroyAllWindows()
        return

    # 5) Seleziona contorno più grande (area) e disegna bounding box
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    if area < min_area:
        print(f"❌ Nessun contorno abbastanza grande. Area max: {area}")
        cv2.destroyAllWindows()
        return

    x, y, w, h = cv2.boundingRect(main_contour)
    boxed = img.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("4 - Bounding Box", boxed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 🧪 Esempi
detect_main_object_with_canny_tuning("./images/Mele-altasfera.jpg")
detect_main_object_with_canny_tuning("../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg")
