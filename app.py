from flask import Flask, jsonify, request
from flask_cors import CORS


from PIL import Image
import numpy as np
import json
import base64
from io import BytesIO, StringIO
import time
import requests
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


from util.eggmodel import get_eggmodel
from util.np_encoder import NpEncoder
from datetime import datetime

app = Flask(__name__)
CORS(app)

eggmodel = get_eggmodel('data\kita_coba.onnx')
print("eggmodel", eggmodel)


@app.route("/")
def index():
    return jsonify({
        'message': 'welcome to eggmodel-be-ml'
    }), 200


@app.route("/egg-category", methods=['POST'])
def predict_egg_category():
    image = None

    if 'image-type' in request.args.to_dict() and request.args['image-type'] == 'b64':
        print('using base64')
        body = request.json
        print(body.keys())
        if 'image' not in request.json:
            response = jsonify(json.loads(json.dumps(
                {'message': 'No image'}, cls=NpEncoder)))
            response.status_code = 400
            response.headers['Content-Type'] = 'application/json'
            return response


        # pIndex = body['image'].find("</p>")
        # if pIndex != -1:
        #     body['image'] = body['image'][pIndex+4:]

        # pIndex = body['image'].find("?image-type=b64")
        # if pIndex != -1:
        #     body['image'] = body['image'][pIndex+15:]

        # pIndex = body['image'].find("}")
        # if pIndex != -1:
        #     body['image'] = body['image'][pIndex+1:]

        pIndex = body['image'].find("/9j/")
        if pIndex != -1:
            body['image'] = body['image'][pIndex:]

        base64String = body['image']
        print("img:",base64String)

        # im_bytes is a binary image
        im_bytes = base64.b64decode(base64String)
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        image = Image.open(im_file)   # img is now PIL Image object

    else:
        print('using file image')
        if 'image' not in request.files:
            response = jsonify(json.loads(json.dumps(
                {'message': 'No image'}, cls=NpEncoder)))
            response.status_code = 400
            response.headers['Content-Type'] = 'application/json'
            return response

        file = request.files['image']
        image = Image.open(file)

    image = image.resize((224, 224), Image.LANCZOS)

    image = np.array(image)
     # Convert the image data to float in the range [0, 1]
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    inference_time_start = time.time()
    output = eggmodel.run(None, {'input_1': image})
    inference_time_stop = time.time()
    inference_duration = inference_time_stop - inference_time_start
    classnames = ['1',
                  '2',]

    output_index = np.argmax(output)
    output_classname = classnames[output_index]
    current_date = datetime.now().isoformat() 

    

    response = jsonify(json.loads(json.dumps({
        'classIndex': output_index,
        'className': output_classname,
        'inferenceTimeSeconds': inference_duration,
        'img': base64String,
        'time': current_date,
    }, cls=NpEncoder)))
    response.status_code = 200
    return response