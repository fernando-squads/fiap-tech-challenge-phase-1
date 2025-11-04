# Biblioteca para criar aplicativos web e APIs
from flask import Flask
# Biblioteca para simplificar e acelerar a construção de APIs RESTful. Ele adiciona um conjunto de ferramentas e decoradores para ajudar a descrever e documentar a API de forma automática
from flask_restx import Api, Resource, fields
# Biblioteca para otimizar e acelerar fluxos de trabalho que envolvem tarefas computacionalmente intensivas
import joblib
# Biblioteca para trabalhar com arrays multidimensionais e matrizes de forma eficiente
import numpy as np
# Biblioteca para importar arquivos
import pandas as pd
from waitress import serve

# Carregar modelo e scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)
api = Api(app, version='1.0', title='Diabetes API',
          description='API to check if a person has diabetes using artificial intelligence.')

# Define a namespace
ns = api.namespace('Diabetes', description='Diabetes operations')

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
prediction_model = api.model('Prediction', {
    'hasDiabetes': fields.Boolean(readOnly=True, description='Express the prediction'),
    'probability': fields.String(required=True, description='Express the probability')
})


@ns.route('/')
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

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8090)