import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from utils import load_pickle, make_prediction, process_label, process_json_csv, output_batch, return_columns
import pandas as pd
import pickle
from contextlib import asynccontextmanager
from typing import List
from pydantic import BaseModel
from mangum import Mangum


model = load_pickle("./artifacts/model.pkl")


# Input Data Validation
class ModelInput(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int


# Input for Batch prediction
class ModelInputs(BaseModel):
    all: List[ModelInput]

    def return_dict_inputs(cls):
        return [dict(input) for input in cls.all]
    


# model prediction
def diabetes_predictor(x: dict) -> dict:
    with open("./artifacts/model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)

    data_df = pd.DataFrame(x, index=[0])
    prediction = model.predict(data_df)

    if(prediction[0]>0.5):
        prediction="Diabetic"
    else:
        prediction="Non Diabetic"
    
    return {'prediction': prediction}


# Life Span Management
ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["diabetes_predictor"] = diabetes_predictor
    yield
    ml_models.clear()


# Create an instance of FastAPI
app = FastAPI(lifespan=ml_lifespan_manager, debug=True)

# Mangum wrapper for AWS deployment
handler = Mangum(app)

# Root endpoint
@app.get('/')
def index():
    return {'message': 'Welcome to the Diabetes API'}


# Health check endpoint
@app.get("/health")
def check_health():
    return {"status": "ok"}


# Model information endpoint
@app.post('/model-info')
async def model_info():
    model_name = ml_models["diabetes_predictor"].__class__.__name__ # get model name 
    model_params = ml_models["diabetes_predictor"].get_params() # get model parameters

    # model_name = model.__class__.__name__ # get model name 
    # model_params = model.get_params() # get model parameters

    model_information =  {'model info': {
            'model name ': model_name,
            'model parameters': model_params
            }
            }
    return model_information # return model information


# Single Prediction endpoint 
@app.post("/predict")
async def predict(model_input: ModelInput):

    data = dict(model_input)

    return ml_models["diabetes_predictor"](data)


# Batch prediction endpoint
@app.post('/predict-batch')
async def predict_batch(inputs: ModelInputs):
    # Create a dataframe from inputs
    data = pd.DataFrame(inputs.return_dict_inputs())
    labels, probs = make_prediction(data, model) # Get the labels
    response = output_batch(data, labels) # output results
    return response



# Upload data endpoint
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    file_type = file.content_type # get the type of the uploaded file
    valid_formats = ['text/csv', 'application/json'] # create a list of valid formats API can receive
    if file_type not in valid_formats:
        return JSONResponse(content={"error": f"Invalid file format. Must be one of: {', '.join(valid_formats)}"}) # return an error if file type is not included in the valid formats
    
    else:
        contents = await file.read() # read contents in file
        data= process_json_csv(contents=contents,file_type=file_type, valid_formats=valid_formats) # process files  
        labels, probs = make_prediction(data, model) # Get the labels
        response = output_batch(data, labels) # output results

    return response


# Run the FastAPI application
# http://127.0.0.1:8000/predict
if __name__ == '__main__':
    uvicorn.run('app:app', reload=True)