from fastapi import FastAPI
import pickle
from pydantic import BaseModel
# import uvicorn
from os import getenv
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
async def predict_diabetes(data: InputData):
    # Convert input data to a list and reshape it
    input_data = [list(data.model_dump().values())]
    
    # Make prediction
    prediction = diabetes_model.predict(input_data)[0]
    
    return {"prediction": "hello"}



# if __name__ == "__main__":
#     port = int(getenv("PORT",8000))
#     uvicorn.run("app.api:app",host="0.0.0.0",port=port,reload=True)
