import logging
import pandas as pd
import os


def test_data_shape(data):
    """
    Test shape of the data
    """
    # Check the df shape
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing dataset: The file doesn't appear to have rows and columns")
        raise err


def test_data_features(data, features):
    """
    Test features of the data
    """
    try:

        assert set(data.columns) == set(features)

    except AssertionError as err:
        logging.error(
            "Testing dataset: Features are missing in the data columns")
        raise err


def test_model(model, dataset_split):
    """
    Check if model is able to make predictions
    """
    try:
        X_train, y_train, X_test, y_test = dataset_split
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    except Exception as err:
        logging.error(
            "Testing model: Saved model is not able to make new predictions")
        raise err
