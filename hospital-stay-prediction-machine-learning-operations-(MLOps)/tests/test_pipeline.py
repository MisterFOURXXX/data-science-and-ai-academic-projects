import pytest
import numpy as np
import joblib
import json
import os

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

def test_evaluation_metrics_exist():
    """Test that evaluation.json file exists after evaluation stage"""
    assert os.path.exists("metrics/evaluation.json"), "evaluation.json not found in metrics/ directory"

def test_evaluation_metrics_valid():
    """Test that evaluation metrics are valid and within expected ranges"""
    with open("metrics/evaluation.json") as f:
        metrics = json.load(f)
    
    required_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_roc_macro"]
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert isinstance(metrics[metric], (int, float)), f"{metric} is not a number"
        assert 0 <= metrics[metric] <= 1, f"{metric} value {metrics[metric]} is out of range [0, 1]"

def test_model_consistency():
    """Test that model predictions are deterministic for the same input"""
    model = joblib.load("models/final_stacking_model.pkl")
    X_test = np.load("data/processed/X_test.npy")
    
    # Load model again to ensure consistency
    model_reload = joblib.load("models/final_stacking_model.pkl")
    
    # Get predictions from both model instances
    pred1 = model.predict(X_test[:10])
    pred2 = model_reload.predict(X_test[:10])
    
    np.testing.assert_array_equal(pred1, pred2, 
                                   err_msg="Model predictions are not consistent across different loads")

def test_model_probability_consistency():
    """Test that predict_proba produces consistent probabilities"""
    model = joblib.load("models/final_stacking_model.pkl")
    X_test = np.load("data/processed/X_test.npy")
    
    proba1 = model.predict_proba(X_test[:5])
    proba2 = model.predict_proba(X_test[:5])
    
    np.testing.assert_array_almost_equal(proba1, proba2, decimal=10,
                                         err_msg="Model probabilities are not consistent")

def test_weak_learner_models_exist():
    """Test that all weak learner models were saved correctly"""
    weak_learner_names = ["lgb_model.pkl", "xgb_model.pkl", "cat_model.pkl"]
    for model_name in weak_learner_names:
        model_path = f"models/{model_name}"
        assert os.path.exists(model_path), f"Weak learner model not found: {model_path}"
        model = joblib.load(model_path)
        assert model is not None, f"Failed to load {model_path}"

def test_preprocessor_files_exist():
    """Test that all preprocessor files are saved"""
    preprocessor_files = ["preprocessor.pkl", "pca.pkl", "label_encoder.pkl"]
    for file_name in preprocessor_files:
        file_path = f"models/{file_name}"
        assert os.path.exists(file_path), f"Preprocessor file not found: {file_path}"

def test_label_encoder_consistency():
    """Test that label encoder is consistent for encoding/decoding"""
    le = joblib.load("models/label_encoder.pkl")
    y_test = np.load("data/processed/y_test.npy")
    
    # Get unique classes
    classes = le.classes_
    assert len(classes) > 0, "Label encoder has no classes"
    
    # Test inverse transform
    original_labels = le.inverse_transform(y_test[:10])
    assert len(original_labels) == 10, "Inverse transform failed"
    
    # Verify round-trip encoding/decoding
    encoded = le.transform(original_labels)
    np.testing.assert_array_equal(encoded, y_test[:10],
                                   err_msg="Label encoder round-trip transform failed")