import cv2
import numpy as np

def hsv_sobel_laplace_bbox(img, v_thresh=180, sobel_thresh=50, laplace_thresh=20):
    h, w = img.shape[:2]
    
    # 1. Maschera HSV sul canale V
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, v_mask = cv2.threshold(hsv[:, :, 2], v_thresh, 255, cv2.THRESH_BINARY)
    
    # 2. Gradiente Sobel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    abs_sobelx = cv2.convertScaleAbs(sobelx)    # valore assoluto X
    abs_sobely = cv2.convertScaleAbs(sobely)    # valore assoluto Y
    sobel_mag = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    _, sobel_mask = cv2.threshold(sobel_mag, sobel_thresh, 255, cv2.THRESH_BINARY)
    
    # 3. Laplaciano
    lap = cv2.Laplacian(gray, cv2.CV_16S)
    lap_mag = cv2.convertScaleAbs(lap)
    _, lap_mask = cv2.threshold(lap_mag, laplace_thresh, 255, cv2.THRESH_BINARY)
    
    # 4. Combina maschere
    combined = cv2.bitwise_and(v_mask, sobel_mask)
    combined = cv2.bitwise_or(combined, lap_mask)
    combined = cv2.medianBlur(combined, 5)
    
    # 5. Contorni e bounding box
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        x, y, bw, bh = w//4, h//4, w//2, h//2
        fallback = np.zeros((h, w), np.uint8)
        fallback[y:y+bh, x:x+bw] = 255
        return fallback, (x, y, bw, bh)
    
    main = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(main)
    final_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(final_mask, [main], -1, 255, -1)
    
    return final_mask, (x, y, bw, bh)



def main():
    #img_path = "../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg"
    img_path ="../Holopix50k/val/left/-L__lwrICsexa0oiALCB_left.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))
    
    mask, (x, y, w_box, h_box) = hsv_sobel_laplace_bbox(img,
                                                        v_thresh=170,
                                                        sobel_thresh=60,
                                                        laplace_thresh=15)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    cv2.rectangle(img, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
    
    cv2.imshow("Bounding Box", img)
    cv2.imshow("Segmented", segmented)
    cv2.imshow("Mask", mask)
    print(f"Box: x={x}, y={y}, w={w_box}, h={h_box}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
