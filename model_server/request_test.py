import requests
from skimage import io
import json
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from skimage.segmentation import felzenszwalb
from skimage.segmentation import *
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
# urllib.request.urlretrieve("https://i.imgur.com/RzdqHl9.png", "malignant.jpg")
resp = requests.post("http://localhost:5000/predict",
                     files={"file":open('assets/benign (103).png','rb')})
# resp = requests.post("http://localhost:5000/predict",
#                      files={"file":open('malignant.jpg','rb')})

json_load= resp.json()
a_restored = np.asarray(json_load["mask"])
io.imshow(a_restored)
io.show()