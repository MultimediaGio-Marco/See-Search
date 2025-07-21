# 👀 Stereo Object Detection Pipeline

Pipeline completa per l'elaborazione di immagini **stereo**, mirata a rilevare **oggetti salienti** tramite edge detection, stima della profondità e fusione delle mappe di salienza.

---

## 📊 Schema della Pipeline

![Schema Pipeline](https://raw.githubusercontent.com/MultimediaGio-Marco/immaginiGradoTD/main/pipeline_integrazione_deaphMap.png)

> Ogni nodo mostra una fase reale della pipeline: da input stereo a maschere finali e box rilevati.

---

## 🧱 Architettura Modulare

La pipeline è divisa in tre moduli principali:

* `StereoEdgeProcessor`: Estrae edge salienti dalle immagini stereo.
* `DeapMapProcessor`: Calcola la profondità e genera maschere di salienza 3D.
* `StereoObjectDetector`: Coordina l'esecuzione parallela e fonde i risultati per individuare box affidabili.

---

## 🔄 Flusso della Pipeline

### 1. 🔧 Preprocessing (interno ai moduli)

Le immagini stereo vengono preprocessate:

* Conversione in **YCrCb**
* Equalizzazione del canale Y
* Calcolo di edge: **Canny**, **Sobel Magnitude**, **Laplaciano**

---

### 2. 🧠 Estrazione Edge Stereo

Il modulo `StereoEdgeProcessor` esegue edge detection su **entrambe** le immagini e ne fonde i risultati:

```python
results = stereo_edge_enhanced(left_img, right_img)
edge_map = results["final"]
```

---

### 3. 🌊 Stima della Profondità

Il modulo `DeapMapProcessor` calcola:

* Mappa di profondità con **StereoSGBM**
* Maschera di salienza 3D
* Visualizzazione con colormap `inferno`

```python
depth_map, depth_color, depth_mask = depth_processor.process()
```

---

### 4. 🔀 Fusione Edge & Profondità

Le due maschere (`depth_mask` e `edge_mask`) vengono combinate per evidenziare le regioni veramente salienti:

```python
combined_mask = cv2.bitwise_and(depth_mask, edge_mask)
```

---

### 5. 📦 Estrazione Box

Dalla maschera combinata si estraggono le **bounding box**:

```python
contours, _ = cv2.findContours(combined_mask, ...)
boxes = [cv2.boundingRect(c) for c in contours if area > min_area]
```

---

### 6. 🧹 Pulizia e Refinement

Le box vengono:

* Clusterizzate con **DBSCAN**
* Filtrate per rimuovere quelle **troppo grandi**
* Escluse le **box contenute** in altre

```python
merged_bboxes, _ = cluster_boxes_dbscan(...)
notobig = remove_bigest_boxes(merged_bboxes, img_area, 0.9)
noinside = remove_contained_boxes(notobig)
```

---

### 7. 🎯 Selezione Best Box

Viene scelta la **miglior box** combinando:

* Mappa di profondità
* Mappa di edge

```python
best_box = pick_best_box(noinside, edge_mask, depth_map)
```

---

## ⚙️ Come Eseguire

Puoi usare direttamente la classe principale:

```python
from stereo_detector import StereoObjectDetector

detector = StereoObjectDetector(left_path, right_path)
result = detector.run()
```

Tutti i moduli vengono eseguiti in parallelo, e il risultato include:

* Mappa di profondità
* Maschere di edge
* Maschera combinata
* Box grezzi, box raffinati
* Miglior box

---

## 🖼️ Visualizzazione

```python
cv2.imshow("Depth Mask", result["depth_mask"])
cv2.imshow("Edge Mask", result["edge_mask"])
cv2.imshow("Combined Mask", result["combined_mask"])
cv2.imshow("Detected Boxes", out_img)
```

---

## 📁 Dataset di riferimento

Testato su immagini stereo del dataset **Holopix50k**.

---

## 👥 Autori

* **Giovanni Oliverio**
* **Marco D'Albis**
