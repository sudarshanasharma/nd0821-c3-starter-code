"""
Author: Sudarshana Sharma
Date: June, 2024
This is the  main file for our Application
"""
from fastapi import FastAPI, Response, status
from contextlib import asynccontextmanager
from ml.model import load_model, predict_single
from pydantic import BaseModel, Field
import logging
import os
import uvicorn


logging.basicConfig(level=logging.INFO)
MODEL_PATH = "./model"
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -df datastore s3://udacity-sudarshana-mlops-4")
    if os.system("dvc pull -f") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")


class Data(BaseModel):
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias="native-country")
    age: int
    fnlwgt: int
    education_num: int  = Field(alias="education-num")
    capital_gain: int  = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")

    class Config:
        allow_population_by_field_name = True,
        json_schema_extra = {
            "example": {
                "workclass": "state_gov",
                "education": "bachelors",
                "marital_status": "never_married",
                "occupation": "adm_clerical",
                "relationship": "not_in_family",
                "race": "white",
                "sex": "male",
                "native_country": "united_states",
                "age": 39,
                "fnlwgt": 77516,
                "education_num": 13,
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40
            }
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading model")
    global model, encoder, lb
    # if the model exists, load the model
    print(os.path.isfile(os.path.join(MODEL_PATH, 'model.pkl')))
    if os.path.isfile(os.path.join(MODEL_PATH, 'model.pkl')):
        model, encoder, lb = load_model(MODEL_PATH)
    logging.info("Model loaded")
    yield

# Instantiate the app.
app = FastAPI(lifespan=lifespan)


# welcome message on the root
@app.get("/")
def read_root():
    response = Response(
        status_code=status.HTTP_200_OK,
        content="Welcome to Udacity Income Prediction API"
    )
    return response


# model inference:
@app.post("/predict")
def predict(data: Data):
    # if any data same as example, return error
    logging.info(f"data dict: {data.dict().values()}")

    # Check if any string data is missing:
    if 'string' in data.dict().values():
        response = Response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content="Please enter all the data correctly"
        )
        return response

    else:
        logging.info("Model inference started")
        y_pred = predict_single(data, MODEL_PATH)
        logging.info("Prediction completed")

        response = Response(
            status_code=status.HTTP_200_OK,
            content="The predicted income is: " + str(list(y_pred)[0]),
        )

        return response


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))