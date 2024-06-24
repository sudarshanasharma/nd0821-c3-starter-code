import requests

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
print(f"Status Code: {post_response.status_code}")
print(f"Model Inference: {post_response.content}")
