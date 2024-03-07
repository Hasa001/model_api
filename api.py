from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import json
# import uvicorn
# from os import getenv
# Load the model
# with open('app/diabetes_model.sav', 'rb') as file:
#     diabetes_model = pickle.load(file)

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))
# Define the FastAPI app
app = FastAPI()

# Define a Pydantic model to represent the input data
class InputData(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int

# Define the route to handle POST requests
@app.post("/predict/")
async def predict_diabetes(input_parameters: InputData):
    # # Convert input data to a list and reshape it
    # input_data = [list(data.model_dump().values())]
    # print("working")
    # # Make prediction
    # prediction = diabetes_model.predict(input_data)[0]
    
    # return {"prediction": prediction}
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']


    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list])
    
    if prediction[0] == 0:
        return 'The person is not Diabetic'
    
    else:
        return 'The person is Diabetic'



# if __name__ == "__main__":
#     port = int(getenv("PORT",8000))
#     uvicorn.run("app.api:app",host="0.0.0.0",port=port,reload=True)
