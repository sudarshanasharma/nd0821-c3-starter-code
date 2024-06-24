from ml.data import clean_data, process_data
from ml.model import train_model, compute_model_metrics, inference


def test_cleaning(raw_data):
    cleaned_data, cat_cols, num_cols = clean_data(raw_data, "/tmp/census_cleaned.csv", "salary")
    assert cleaned_data.shape == raw_data.shape
    assert cleaned_data.isna().sum().sum() == 0
    assert len(cat_cols) == 8
    assert len(num_cols) == 6


def test_process_data(data, cat_features):
    X_train, y_train, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert X_train.shape == (32561, 105)
    assert y_train.shape == (32561,)
    assert encoder is not None
    assert lb is not None


def test_train_model(dataset_split):
    X_train = dataset_split[0]
    y_train = dataset_split[1]
    model = train_model(X_train, y_train)
    assert model is not None


def test_compute_model_metrics(model, dataset_split):
    y_pred = inference(model, dataset_split[2])
    precision, recall, fbeta = compute_model_metrics(dataset_split[3], y_pred)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference(model, dataset_split):
    X_test = dataset_split[2]
    y_pred = inference(model, X_test)
    assert y_pred is not None
