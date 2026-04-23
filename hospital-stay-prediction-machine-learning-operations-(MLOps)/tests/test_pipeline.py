import pytest
import numpy as np
import joblib

def test_data_shapes():
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    assert X_train.shape[0] > 0
    assert len(y_train) == X_train.shape[0]

def test_preprocessor():
    preprocessor = joblib.load("models/preprocessor.pkl")
    assert preprocessor is not None

def test_model_prediction():
    model = joblib.load("models/final_stacking_model.pkl")
    X_test = np.load("data/processed/X_test.npy")
    pred = model.predict(X_test[:5])
    assert len(pred) == 5