"""
Author: Sudarshana Sharma
Date: June, 2024
This script is used to post to the API using the requests module
and returns both the result of model inference and the status code
"""
import logging
import requests

logging.basicConfig(level=logging.INFO)

def main():
    # Post response from the server
    sample_dict = {'workclass': 'state_gov',
                   'education': 'bachelors',
                   'marital_status': 'never_married',
                   'occupation': 'adm_clerical',
                   'relationship': 'not_in_family',
                   'race': 'white',
                   'sex': 'male',
                   'native_country': 'united_states',
                   'age': 39,
                   'fnlwgt': 77516,
                   'education_num': 13,
                   'capital_gain': 2174,
                   'capital_loss': 0,
                   'hours_per_week': 40
                   }
    url = "https://udacity-ssharma-income-predict-5eaa88cf7b4f.herokuapp.com/predict"
    post_response = requests.post(url, json=sample_dict)
    status_code = post_response.status_code
    content_text = post_response.content
    logging.info(f"Status Code: {status_code}")
    logging.info(f"Model Inference: {content_text}")
    return status_code, content_text

if __name__ == "__main__":
    main()
