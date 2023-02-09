from flask import Flask, render_template, request, jsonify
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS, cross_origin
from PIL import Image
import io

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = tf.keras.models.load_model('model.h5')
classifier = load_model('model.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
@cross_origin()
def main():
    return 'Welcome to TB Lab API'

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    pred = np.argmax(model.predict(img))

    if pred==0:
        return 'NORMAL'
    elif pred==1:
        return 'TUBERCULOSIS'
 
def predict_percentage(img):
    predict_proba = sorted(model.predict(img)[0])[1]
    return round(predict_proba*100,2),

@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    if 'imagefile' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    imagefile = request.files.get('imagefile')

    if not imagefile:
        return

    img_bytes = imagefile.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img), probability=predict_percentage(img)[0])

if __name__ == '__main__':
    app.run(port=3000, debug=True)