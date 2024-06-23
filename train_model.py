"""
Author: Sudarshana Sharma
Date: June, 2024
This script is used to train a machine learning model
"""
# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.model import *
import logging
import os
import warnings

warnings.filterwarnings("ignore")

# Initialize logging
logging.basicConfig(level=logging.INFO)


def main():
    logging.info(12 * '*' + 'STARTED MODEL TRAINING' + 12 * '*')
    # Load data
    logging.info("Loading data...")
    data = pd.read_csv("data/census.csv")

    # Clean data
    cleaned_data, cat_cols, num_cols = clean_data(
        data, "data/census_cleaned.csv", "salary")

    # split data into train and test
    train, test = train_test_split(
        cleaned_data, test_size=0.20, random_state=107, stratify=data['salary'])

    logging.info(
        f"Data shape after split. Train: {train.shape} \t Test:{test.shape}")

    # Process the training data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_cols, label="salary", training=True)

    # Proces the TEST data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_cols, label="salary", training=False, encoder=encoder, lb=lb)

    logging.info(f"X_test data shape after transformation: {X_test.shape}")

    # Train and save a model.
    MODEL_PATH = './model'

    # if the model exists, load the model
    if os.path.isfile(os.path.join(MODEL_PATH, 'model.pkl')):
        logging.info(f"A model already exists...")
        model, encoder, lb = load_model(MODEL_PATH)
        logging.info(f"model, encoder and labeler loaded")

    else:
        logging.info(f"A model does not exist... finding the best model...")
        model = train_model(X_train, y_train)
        save_model(model, MODEL_PATH, encoder, lb)
        logging.info(f"Best model saved")

    # Evaluate the model on the test data.
    y_pred = inference(model, X_test)
    # Compute the model metrics on the test data.
    logging.info("### Test metrics ###")
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    logging.info(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")

    # compute slice_feature (feature = education) metrics
    slice_metrics = compute_slice_metrics(
        test, 'salary', cat_cols, 'education', model, encoder, lb)

    logging.info(12 * '*' + 'FINISHED MODEL TRAINING ' + 12 * '*')


if __name__ == '__main__':
    main()
