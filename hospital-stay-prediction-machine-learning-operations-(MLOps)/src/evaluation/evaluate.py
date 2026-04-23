import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import mlflow
from src.utils.mlflow_utils import init_mlflow
import os

init_mlflow()

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")
model = joblib.load("models/final_stacking_model.pkl")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
    "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    "auc_roc_macro": float(roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro"))
}

with mlflow.start_run(run_name="final_evaluation"):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        
os.makedirs("metrics", exist_ok=True)

with open("metrics/evaluation.json", "w") as f:
    json.dump(metrics, f)