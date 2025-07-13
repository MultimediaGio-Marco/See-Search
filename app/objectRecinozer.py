from ilPiùFunzionale import grabcut_bounding_box
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import requests

class ObjectRecognizer():
    
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        # Trasformazioni immagine per ImageNet
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

            # Carica le label di ImageNet
        response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
        self.labels = response.text.strip().split('\n')

    def recognize(self,img_path):
        """
        Riconosce oggetti in un'immagine utilizzando MobileNetV2 e GrabCut per il bounding box.
        Restituisce la maschera binaria e le etichette degli oggetti riconosciuti.
        """
        # Carica l'immagine e ridimensionala
        img = cv2.imread(img_path)
        final_mask, (x, y, w_box, h_box) = grabcut_bounding_box(img)
        oggettoRitagliato = img[y:y+h_box, x:x+w_box]
        self.recognize_object_torch(oggettoRitagliato)
        cv2.imshow("Immagine originale", img)
        cv2.imshow("Oggetto ritagliato", oggettoRitagliato)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        

    def recognize_object_torch(self,image_bgr):
            # Converti da OpenCV (BGR) a PIL (RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

            # Preprocessa immagine
        input_tensor = self.transform(image_pil).unsqueeze(0)

            # Predizione
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Primi 3 risultati
        top3 = torch.topk(probabilities, 3)
        for i in range(3):
            idx = top3.indices[i].item()
            prob = top3.values[i].item()
            print(f"{i+1}: {self.labels[idx]} ({prob * 100:.2f}%)")

if __name__ == "__main__":
    recognizer = ObjectRecognizer()
    recognizer.recognize("../Holopix50k/val/left/-LahPsJhCZTWwgvaAMB4_left.jpg")

            