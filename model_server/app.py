import os
import io
import skimage.io as sio
import json
from utils import *
from PIL import Image
from flask import Flask,request
import tensorflow as tf
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, find_boundaries
from skimage.transform import resize
from skimage.color import rgb2gray
app = Flask(__name__)
inputs = tf.keras.layers.Input((256, 256, 3))
model_ultron = UNet(inputs, dropouts=0.7)
model_ultron.load_weights('breast_cancer_segmentation_inference.h5')

model_classify =  define_model()
model_classify.load_weights('BreastCancerClassifer.h5')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/')
def index():
    return 'Web App with Python Flask! Hey'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file = np.array(Image.open(io.BytesIO(file.read())))
        shape = 256
        file =  resize(file, (shape,shape))
        img = []
        for i in range(1,16):
            img.append(file)
        file = np.array(img[0:16])
        output= model_ultron.predict(file)
        output = output[1][:,:,0]
        output[output < 0.7] = 0
        output[output  >= 0.7] = 1
        segments_fz = np.array(felzenszwalb(output, scale=3.0, sigma=0.95, min_size=600),dtype=np.uint8)
        boundaries = find_boundaries(segments_fz, mode='outer').astype(np.uint8)
        return json.dumps({'mask':boundaries}, cls=NumpyEncoder)
    
@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        file = request.files['file']
        file = np.array(Image.open(io.BytesIO(file.read())))
        shape = 128
        file =  rgb2gray(resize(file, (shape,shape)))
        file = file.reshape(shape,shape,1)
        file = np.array([file])
        output= np.argmax(model_classify.predict(file))
        return json.dumps({'class': int(output)})
        

if __name__ == '__main__':
    app.run()