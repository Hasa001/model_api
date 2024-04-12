from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class diabetes_model_input(BaseModel):
    Pregnancies : int
    Glucose : float
    BloodPressure : float
    SkinThickness : float
    Insulin : float
    BMI : float
    DiabetesPedigreeFunction :  float
    Age : int

class heart_model_input(BaseModel):

    age	: int
    sex	: int
    cp	: float
    trestbps	: float
    chol	: float
    fbs	: float
    restecg	: float
    thalach	: float
    exang	: float
    oldpeak	: float
    slope	: float
    ca	: float
    thal: float


   
# loading the saved model
diabetes_model = pickle.load(open('Diabetes_Model.sav','rb'))
heart_health_model = pickle.load(open('Heart_Disease_Model.sav','rb'))
stress_model = pickle.load(open('Stress_Prediction_Model.sav','rb'))

scaler = pickle.load(open('scaler.sav', 'rb'))
scaler2= pickle.load(open('scaler2.sav','rb'))
scaler3= pickle.load(open('tf_transform.sav','rb'))

@app.post('/diabetes_prediction')
async def diabetes_pred(input_parameters : diabetes_model_input):
    preg=input_parameters.Pregnancies
    glu = input_parameters.Glucose
    bp = input_parameters.BloodPressure
    skin = input_parameters.SkinThickness
    insulin = input_parameters.Insulin
    bmi = input_parameters.BMI
    dpf = input_parameters.DiabetesPedigreeFunction
    age = input_parameters.Age


    input_array = np.array([[preg,glu, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_array)
    prediction = diabetes_model.predict(input_scaled)[0] 
    if prediction == 1:
        return {'The person is Diabetic'}
    
    else:
        return {'The person is not Diabetic'}

@app.post('/heart_health_prediction')
async def heart_health_pred(input_params : heart_model_input):

    age	= input_params.age
    sex	= input_params.sex
    cp	= input_params.cp
    trestbps	= input_params.trestbps
    chol	= input_params.chol
    fbs	= input_params.fbs
    restecg	= input_params.restecg
    thalach	= input_params.thalach
    exang	= input_params.exang
    oldpeak	= input_params.oldpeak
    slope	= input_params.slope
    ca	= input_params.ca
    thal= input_params.thal

    input_array = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler2.transform(input_array)
    prediction = heart_health_model.predict(input_scaled)[0] 
    if prediction == 1:
        return {'The person has heart disease'}
    
    else:
        return {'The person doesn\'t have heart disease'}
    
@app.post('/stress_prediction')
async def heart_health_pred(text):

    input_array = np.array([[text]])
    input_scaled = scaler3.transform(input_array)
    prediction = stress_model.predict(input_scaled)[0] 
    if prediction == 1:
        return {'this person is in stress'}
    
    else:
        return {'this person is not in stress'}