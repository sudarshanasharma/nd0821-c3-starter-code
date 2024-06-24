import logging
import os
import pickle
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data, clean_data

logging.basicConfig(level=logging.INFO)


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        random_state=0)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Computes the following metrics on the machine learning model:
    precision, recall, F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path, encoder=None, labeler=None):
    """
    Save the given model object to a specified file path
    using pickle.

    Inputs
    ------
    model : object
        The machine learning model object to be saved.
    path : str
        The file path to save the model to.
    encoder : object, optional
        The encoder object used to transform input data, if any.
    labeler : object, optional
        The label encoder object used to transform target data, if any.
    """
    pickle.dump(model, open(os.path.join(path, 'model.pkl'), 'wb'))
    if encoder:
        pickle.dump(encoder, open(os.path.join(path, 'encoder.pkl'), 'wb'))

    if labeler:
        pickle.dump(labeler, open(os.path.join(path, 'labeler.pkl'), 'wb'))


def load_model(model_dir):
    """
    Load a previously saved machine learning model object,
    encoder object, and labeler object.

    Inputs
    ------
    model_dir : str
        The file path to load the saved model from.

    Returns:
        tuple: A tuple containing the loaded model object, encoder object, and labeler object.
    """
    model_path = os.path.join(model_dir, 'model.pkl')
    model = pickle.load(open(model_path, 'rb'))

    encoder_path = os.path.join(model_dir, 'encoder.pkl')
    encoder = pickle.load(open(encoder_path, 'rb'))

    labeler_path = os.path.join(model_dir, 'labeler.pkl')
    lb = pickle.load(open(labeler_path, 'rb'))

    return model, encoder, lb


def compute_slice_metrics(
        cleaned_df,
        target,
        categorical_features,
        feature,
        model,
        encoder,
        lb):
    """
    Computes the model metrics for each slice of data that
    has a particular value for a given feature.

    Inputs
    ------
    cleaned_df : pd.DataFrame
        Cleaned dataframe.
    target : str
        Name of the target column.
    categorical_features : list[str]
        List of the names of the categorical features.
    feature : str
        Name of the feature to compute the slice metrics for.
    model : object
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    Returns
    -------
    slice_metrics : dict
        Dictionary containing the slice metrics for each slice.
    """
    slice_metrics = {}
    for value in cleaned_df[feature].unique():
        X_slice = cleaned_df[cleaned_df[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            X_slice, categorical_features, label=target, training=False, encoder=encoder, lb=lb)
        preds = inference(model, X_slice)
        logging.info(
            f"shape of preds: {preds.shape} & shape of y_slice: {y_slice.shape}")
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        slice_metrics[value] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta}
        logging.info(
            f"slice metrics for {feature} = {value}: {slice_metrics[value]}")

    # write to slice_output.txt
    with open('slice_output.txt', 'w') as f:
        for key, value in slice_metrics.items():
            f.write(f"{key}: {value}")
            f.write("\n")
    return slice_metrics


def predict_single(input_json, model_dir):
    """ Make a prediction using a trained model.

    Inputs
    ------
    model_dir : str
        Path to the directory containing the trained model, encoder & lb.
    input_json : dict
        Input data in json format.
    Returns
    -------
    preds : str
        Prediction class.
    """
    # Convert the input to a dataframe with same data types as training data.
    input_df = pd.DataFrame(dict(input_json), index=[0])
    logging.info(f"input_df: {input_df}")

    # clean data
    cleaned_df, cat_cols, num_cols = clean_data(
        input_df, "data/census_cleaned.csv", "salary", test=True)

    # load model, encoder, and lb and predict on single json instance
    model, encoder, lb = load_model(model_dir)

    # process data
    X, _, _, _ = process_data(
        cleaned_df, cat_cols, training=False, encoder=encoder, lb=lb)

    # predict
    preds = inference(model, X)
    preds_class = lb.inverse_transform(preds)

    return {preds_class[0]}
