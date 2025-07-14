from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
import numpy as np
from ilPiùFunzionale import grabcut_bounding_box

class ObjectRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()

    def recognize(self, img_path, debug=False):
        img = cv2.imread(img_path)
        final_mask, (x, y, w_box, h_box) = grabcut_bounding_box(img)
        cropped = img[y:y+h_box, x:x+w_box]

        if debug:
            cv2.imshow("Originale", img)
            cv2.imshow("Ritaglio", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Converti ritaglio in PIL e passa al modello
        image_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        return self.caption_image(image_pil)

    def caption_image(self, image_pil):
        inputs = self.processor(image_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
