import cv2
from edge_pipeline import stereo_edge_enhanced,postprocess_edges
import os
from box_utils import *
from deapMap import display_relative_depth_map, relative_depth_map

print("Working directory:", os.getcwd())
nomeImage = "-L__uMAz3k25WltzLheY_left"
nomeImage = nomeImage.replace("_left", "")
left_path = f"./Holopix50k/val/left/{nomeImage}_left.jpg"
right_path = f"./Holopix50k/val/right/{nomeImage}_right.jpg"
left = cv2.imread(left_path)
right = cv2.imread(right_path)

results = stereo_edge_enhanced(left, right)

cv2.imshow("Final Combined", results["final"])
cv2.imshow("Fused Canny", results["fused_canny"])
cv2.imshow("Fused Magnitude", results["fused_magnitude"])
cv2.imshow("Fused Laplacian", results["fused_laplacian"])

cv2.waitKey(0)
cv2.destroyAllWindows()

# Estrai bounding box
edge_map = results["final"]

bboxes = extract_bounding_boxes_from_edges(edge_map, min_area=100, threshold=50)

# Disegna le box originali (verde)
output_img = left.copy()
for (x, y, w, h) in bboxes:
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Bounding Boxes Originali (Verdi)", output_img)
cv2.waitKey(0)

# Clusterizza le bounding box con DBSCAN
merged_bboxes, labels = cluster_boxes_dbscan(bboxes, eps=60, min_samples=1)
output_img = left.copy()
for (x, y, w, h) in merged_bboxes:
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
cv2.imshow("Bounding Boxes Clusterizzate (Rosse)", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_area = left.shape[0] * left.shape[1]
notobig=remove_bigest_boxes(merged_bboxes, img_area, 0.9)
noinside = remove_contained_boxes(notobig)
output_img = left.copy()
for (x, y, w, h) in noinside:
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
cv2.imshow("filtered Boxes (blue)", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
depth_map=relative_depth_map(left_path, right_path)
display_relative_depth_map(depth_map)
edge_mask = (edge_map > 50).astype('uint8') * 255  # stesso threshold usato prima
best_box = pick_best_box(noinside, edge_mask, depth_map)

# Disegna best box in giallo
if best_box:
    x, y, w, h = best_box
    output_img = left.copy()
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow("Best Box (Gialla)", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nessuna box selezionata come migliore.")