# 👀 Stereo Edge Detection Pipeline

Pipeline completa per elaborazione di immagini stereo, mirata a rilevare oggetti salienti tramite edge detection e stima della profondità.

---

## 📊 Schema della Pipeline

![Schema Pipeline](https://raw.githubusercontent.com/MultimediaGio-Marco/immaginiGradoTD/main/pipeline.png)

> Ogni nodo del grafo mostra un'immagine intermedia reale generata dalla pipeline, con input stereo e risultati visivi per ogni fase.

---

## 🔍 Fasi principali

### 1. 🧪 Preprocessing

* L'immagine viene convertita nello spazio colore **YCrCb**.
* Si equalizza l'istogramma del canale Y per migliorare il contrasto.

```python
def preprocess(img):
    cv2.GaussianBlur(img, (5, 5), 0, img)
    Y, Cb, Cr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    Y = cv2.equalizeHist(Y)
    return Y
```

---

### 2. 🧠 Calcolo Edge (Left & Right)

* Per ciascuna immagine pre-processata si calcolano:

  * **Canny**
  * **Sobel** → magnitudine
  * **Laplaciano**

```python
def compute_edges(gray):
    canny = cv2.Canny(gray, 50, 150)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = cv2.convertScaleAbs(lap)
    return canny, mag, lap
```

---

### 3. 🔗 Fusione Stereo

* Le mappe edge sinistra e destra vengono fuse tramite media pixel-per-pixel:

```python
def fuse_stereo_maps(maps_left, maps_right):
    fused = [(l.astype(np.float32) + r.astype(np.float32)) / 2 for l, r in zip(maps_left, maps_right)]
    return [cv2.convertScaleAbs(m) for m in fused]
```

---

### 4. 🔀 Final Fusion

* Si calcola la media tra tutte le mappe fuse:

```python
def final_fusion(fused_maps):
    stacked = np.stack(fused_maps, axis=0).astype(np.float32)
    mean_map = np.mean(stacked, axis=0)
    return cv2.convertScaleAbs(mean_map)
```

---

### 5. 📦 Bounding Box

* Si estraggono le box dalla mappa `final`:

```python
bboxes = extract_bounding_boxes_from_edges(edge_map, min_area=100, threshold=50)
```

* Le box vengono clusterizzate con **DBSCAN**

```python
clustered, labels = cluster_boxes_dbscan(bboxes, eps=60, min_samples=1)
```

* Vengono rimosse box troppo grandi o contenute:

```python
notobig = remove_bigest_boxes(clustered, img_area, 0.9)
noinside = remove_contained_boxes(notobig)
```

---

### 6. 📷 Depth Estimation

* Calcolo della mappa di profondità tramite StereoSGBM:

```python
depth_map = relative_depth_map(left_path, right_path)
```

* Visualizzazione con colormap inferno:

```python
plt.imsave("depth_map.png", depth_map, cmap='inferno')
```

---

### 7. 🎯 Selezione Miglior Box

* Si crea una maschera dagli edge con threshold:

```python
edge_mask = (edge_map > 50).astype('uint8') * 255
```

* Si seleziona la best box in base a edge e profondità:

```python
best_box = pick_best_box(noinside, edge_mask, depth_map)
```

---

## ⚙️ Esecuzione

Puoi usare la funzione principale:

```python
from edge_pipeline import stereo_edge_enhanced
```

e collegare i moduli `box_utils` e `deapMap` per completare la pipeline end-to-end.

---

## 🖼️ Visualizzazione

Tutte le immagini usate nel grafo sono salvate nella repository pubblica:

🔗 [`MultimediaGio-Marco/immaginiGradoTD`](https://github.com/MultimediaGio-Marco/immaginiGradoTD)

---

## 👥 Autori

* Giovanni Oliverio
* Marco \[inserisci cognome o nick se vuoi]
