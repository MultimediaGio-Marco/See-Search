from skimage import data
from PIL import Image

img = data.astronaut()   # oppure data.coffee()
Image.fromarray(img).save("sample.png")
print("Salvata sample.png")
