
import os
# Biblioteca para criar aplicativos web e APIs
from flask import Flask
# Biblioteca para simplificar e acelerar a construção de APIs RESTful. Ele adiciona um conjunto de ferramentas e decoradores para ajudar a descrever e documentar a API de forma automática
from flask_restx import Api, Resource, fields, reqparse

from werkzeug.datastructures import FileStorage

from werkzeug.utils import secure_filename
# Biblioteca para otimizar e acelerar fluxos de trabalho que envolvem tarefas computacionalmente intensivas
import joblib

from keras.models import load_model


from PIL import Image

import io
# Biblioteca para trabalhar com arrays multidimensionais e matrizes de forma eficiente
import numpy as np
# Biblioteca para importar arquivos
import pandas as pd
from waitress import serve

# Carregar modelo e scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
pneumonia_model = load_model("pneumonia_model.h5")

app = Flask(__name__)
api = Api(app, version='1.0', title='Predictions API',
          description='API to check if a person has diabetes or pneumonia using artificial intelligence.')

# Define a namespace
ns = api.namespace('predictions', description='Predictions operations')

# Define a model for request body
diabetes_model = api.model('Diabetes', {
    'pregnancies': fields.Integer(readOnly=True, description='Express the Number of pregnancies'),
    'glucose': fields.Integer(required=True, description='Express the Glucose level in blood'),
    'bloodPressure': fields.Integer(required=True, description='Express the Blood pressure measurement'),
    'skinThickness': fields.Integer(required=True, description='Express the thickness of the skin'),
    'insulin': fields.Integer(required=True, description='Express the Insulin level in blood'),
    'bmi': fields.Float(required=True, description='Express the Body mass index'),
    'diabetesPedigreeFunction': fields.Float(required=True, description='Express the Diabetes percentage'),
    'age': fields.Integer(required=True, description='Express the age')
})

# Define a model for response body
prediction_model = api.model('DiabetesPrediction', {
    'hasDiabetes': fields.Boolean(readOnly=True, description='Express the prediction'),
    'probability': fields.String(required=True, description='Express the probability')
})

# Define a model for response body
pneumonia_prediction_model = api.model('PneumoniaPrediction', {
    'result': fields.String(readOnly=True, description='Express the result'),
    'confidence': fields.String(required=True, description='Express the confidence')
})

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file',
                          location='files',
                          type=FileStorage,
                          required=True,
                          help='The file to upload')

IMG_SIZE = (150, 150)

@ns.route('/diabetes')
class DiabetesResource(Resource):
    @ns.doc('Check_Diabetes')
    @ns.expect(diabetes_model)
    @ns.marshal_with(prediction_model, code=200)
    def post(self):
        '''Check Diabetes'''
        data = api.payload
        data_features = [
            data['pregnancies'], data['glucose'], data['bloodPressure'],
            data['skinThickness'], data['insulin'], data['bmi'],
            data['diabetesPedigreeFunction'], data['age']
        ]
        
        features = pd.DataFrame(
            np.array(data_features).reshape(1, -1),
            columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        )
        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        # garantir tipos primitivos (não numpy) para serialização/ marshal
        probability = [float(p) for p in model.predict_proba(features_scaled)[0]]
        response_data = {
            'hasDiabetes': prediction == 1,
            'probability': f"{(probability[prediction] * 100):.2f}%"
        }
        # Retornar dict e status — Flask-RESTX fará o marshal e a serialização
        return response_data, 200

@ns.route('/pneumonia')
class FileUpload(Resource):
    """Handles file uploads and returns a JSON response."""
    @api.expect(upload_parser)
    @ns.marshal_with(pneumonia_prediction_model, code=200)
    def post(self):
        """Uploads a file and processes it."""
        args = upload_parser.parse_args()
        uploaded_file = args['file']

        if uploaded_file:
            # For security, use secure_filename() to sanitize the filename
            filename = secure_filename(uploaded_file.filename)
            
            img = Image.open(io.BytesIO(uploaded_file.read())).convert("L")
            img = img.resize(IMG_SIZE)
            img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Fazer predição
            pred = pneumonia_model.predict(img_arr)[0][0]
            result = "NORMAL" if pred > 0.5 else "PNEUMONIA"
            confidence = f"{(float(pred if pred > 0.5 else 1 - pred) * 100):.2f}%"
            
            # Return a JSON response with a success message
            return {
                'result': result,
                'confidence': confidence
            }, 200 # HTTP status code for created
        
        # If no file is provided, an error will be returned automatically by the parser
        # with an appropriate error message and status code (400 Bad Request).
        return {
            'message': 'File upload failed',
            'status': 'error'
        }, 400

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8090)