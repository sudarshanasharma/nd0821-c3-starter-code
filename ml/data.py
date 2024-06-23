from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def clean_data(df, output_path, target, test=False):
    '''
    Basic cleaning of data
    '''
    logging.info("Cleaning data")

    # remove spaces from column names
    df.columns = df.columns.str.replace(" ", "")

    # filter categorical columns and numerical columns:
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    logging.info(f'Categorical columns: {cat_cols}')
    logging.info(f'Numerical columns: {num_cols}')

    # removing spaces from categorical columns:
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower().str.replace(" ", "")

    # replacing ? with nan:
    logging.info(f'Before replacing ? with nan: {df.isin(["?"]).sum()}')
    df = df.replace('?', np.nan)
    logging.info(f'Nan values: {df.isna().sum()}')

    # fill nan with mode for categorical columns:
    logging.info(f'Filling nan with mode for categorical columns')
    for col in cat_cols:
        if col != target:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            cat_cols.remove(col)

    # fill nan with mean for numerical columns:
    for col in num_cols:
        if col != target:
            df[col] = df[col].fillna(df[col].mean())
        else:
            num_cols.remove(col)
    logging.info(f'After filling nan: {df.isna().sum()}')

    # save cleaned data:
    if not test:
        try:
            df.to_csv(output_path, index=False)
            logging.info(f'Cleaned data saved to {output_path}')
            return df, cat_cols, num_cols
        except BaseException:
            logging.error(f'Unable to save data to {output_path}')
    else:
        logging.info(f'Cleaned data returned')
        return df, cat_cols, num_cols


def process_data(
        X,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
