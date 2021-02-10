from flask import Flask, flash, request, redirect, url_for, session, jsonify
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename
from PIL import Image

import pickle
import numpy as np

# variables Flask
UPLOAD_FOLDER = '/img'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = "mnist"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)

# se carga el modelo de Logistic Regression del Notebook #3
pkl_filename = "ModeloLR.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

pkl_filename2 = "ModeloMNIST.pkl"
with open(pkl_filename2, 'rb') as file2:
    model2 = pickle.load(file2)

@app.route('/predict', methods=['POST'])
def mnist():
    size = 28, 28
    file = request.files['image']
    img = Image.open(file)
    img.thumbnail(size, Image.ANTIALIAS)
    img.save('received-images/img-received.png', "PNG")
    img = np.array(img).reshape(1,-1)

    return jsonify({'Prediccion': int(model2.predict(img)[0])})

@app.route('/predict', methods=['GET'])
def iris():
    # parametros
    parser = reqparse.RequestParser()
    parser.add_argument('petal_length')
    parser.add_argument('petal_width')
    parser.add_argument('sepal_length')
    parser.add_argument('sepal_width')

    # request para el modelo
    args = parser.parse_args() 
    datos = np.fromiter(args.values(), dtype=float) 

    # prediccion
    out = {'Prediccion': int(model.predict([datos])[0])}

    return out, 200

if __name__ == '__main__':
    app.run(debug=True, port='1080')