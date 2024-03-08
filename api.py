from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction :  float
    Age : int
    

# loading the saved model
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))


@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters : model_input):
    
    preg = input_parameters.Pregnancies
    glu = input_parameters.Glucose
    bp = input_parameters.BloodPressure
    skin = input_parameters.SkinThickness
    insulin = input_parameters.Insulin
    bmi = input_parameters.BMI
    dpf = input_parameters.DiabetesPedigreeFunction
    age = input_parameters.Age


    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list])[0] 
    print(prediction)
    if prediction == 1:
        return 'The person is Diabetic'
    
    else:
        return 'The person is not Diabetic'
