from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    """
    Test response from endpoint /
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.content == b'Welcome to Udacity Income Prediction API'


def test_predict_api_negative(data):
    """
    Test negative response from the endpoint /predict
    """
    sample = {
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

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.text == 'The predicted income is: <=50k'


def test_predict_api_positive(data):
    """
    Test positive response from the endpoint /predict
    """
    sample = {
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
            "capital_gain": 10000,
            "capital_loss": 0,
            "hours_per_week": 40
            }

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.text == 'The predicted income is: >50k'


def test_predict_api_invalid():
    """
    Test response for an Invalid request
    """
    data = {}
    response = client.post("/predict", json=data)
    assert response.status_code == 422
