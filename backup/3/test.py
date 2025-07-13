import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from edge_pipeline import edge_pipeline_py  # Cambia con il nome reale del modulo
from find_bounding_boxes import find_bounding_boxes  # Cambia con il nome reale del modulo

def run_full_test(image_path, save_results=True):
    # [1] Carica immagine originale RGB
    image_rgb = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # [2] Applica pipeline per ottenere maschera binaria
    bw_final, combined_edges, denoised_norm = edge_pipeline_py(image_path)

    # [3] Applica bounding box detection
    boxes, contours = find_bounding_boxes(bw_final)

    # [4] Disegna bounding box su copia dell'immagine originale
    image_with_boxes = image_rgb.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # [5] Visualizza i risultati
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Edge Detection + Bounding Boxes", fontsize=16)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(combined_edges, cmap="gray")
    axes[1].set_title("Combined Edges")
    axes[1].axis("off")

    axes[2].imshow(bw_final, cmap="gray")
    axes[2].set_title("Binary Mask")
    axes[2].axis("off")

    axes[3].imshow(image_with_boxes)
    axes[3].set_title(f"Detected Boxes ({len(boxes)})")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    # [6] Salvataggio risultati
    if save_results:
        out_dir = "./test_outputs"
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(os.path.join(out_dir, f"{base_name}_binary_mask.png"), bw_final)
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_edges.png"), combined_edges)
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_with_boxes.png"), cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        print(f"[✔] Risultati salvati in {out_dir}")

if __name__ == "__main__":
    test_image_path = "../Holopix50k/val/left/-L__lwrICsexa0oiALCB_left.jpg"  # <-- cambia con il percorso reale
    run_full_test(test_image_path)
