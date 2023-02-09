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

@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    im = load_img(image_path, target_size=(224,224))
    im_array = np.asarray(im)
    im_array = im_array*(1/224)
    im_input = tf.reshape(im_array, shape = [1, 224, 224, 3])

    pred = np.argmax(model.predict(im_input))
    predict_proba = sorted(model.predict(im_input)[0])[1]
    if pred==0:
        desc = 'NORMAL'
    elif pred==1:
        desc = 'TUBERCULOSIS'

    classification = '%s' % (desc)
    percentage = '%s' % (np.round(predict_proba*100,2))

    return jsonify({'prediction': classification,
                    'probability': percentage})

if __name__ == '__main__':
    app.run(port=3000, debug=True)