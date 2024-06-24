from ml.model import predict_single
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    single_json = {
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
    preds = predict_single(single_json, "./model")
    print(preds)
