import numpy as np
import json
import joblib
import mlflow
import dvc.api
import os
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.utils.mlflow_utils import init_mlflow

os.makedirs("models", exist_ok=True)

params = dvc.api.params_show()
init_mlflow()

X = np.load("data/processed/X_train.npy")
y = np.load("data/processed/y_train.npy")

with open("models/best_weak_params.json") as f:
    best = json.load(f)

weak_learners = [
    ("lgb", LGBMClassifier(**best["LightGBM"], random_state=params["models"]["random_state"], n_jobs=-1, verbose=-1)),
    ("xgb", XGBClassifier(**best["XGBoost"], random_state=params["models"]["random_state"], n_jobs=-1, eval_metric="mlogloss")),
    ("cat", CatBoostClassifier(**best["CatBoost"], random_seed=params["models"]["random_state"], verbose=0))
]

with mlflow.start_run(run_name="final_stacking"):
    skf = StratifiedKFold(
        n_splits=params["models"]["cv"], 
        shuffle=True, 
        random_state=params["models"]["random_state"]
    )
    
    cv_scores = []
    print(f"Starting {params['models']['cv']}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        fold_stacking = StackingClassifier(
            estimators=weak_learners, 
            final_estimator=LogisticRegression(max_iter=1000), 
            cv=3, 
            n_jobs=-1
        )
        
        fold_stacking.fit(X_train_fold, y_train_fold)
        y_pred = fold_stacking.predict(X_val_fold)
        
        fold_f1 = f1_score(y_val_fold, y_pred, average="macro")
        cv_scores.append(fold_f1)
        print(f"Fold {fold+1} Macro F1: {fold_f1:.4f}")

    mean_f1 = np.mean(cv_scores)
    std_f1 = np.std(cv_scores)

    stacking = StackingClassifier(
        estimators=weak_learners, 
        final_estimator=LogisticRegression(max_iter=1000), 
        cv=3, 
        n_jobs=-1
    )
    stacking.fit(X, y)

    mlflow.log_params({
        "cv_folds": params["models"]["cv"],
        "random_state": params["models"]["random_state"],
        "stacking_cv": 3
    })
    
    mlflow.log_metric("mean_macro_f1", mean_f1)
    mlflow.log_metric("std_macro_f1", std_f1)
    
    mlflow.sklearn.log_model(stacking, name="stacking_model")
    
    joblib.dump(stacking, "models/final_stacking_model.pkl", compress=3)
    
    for name, _ in weak_learners:
        trained_base = stacking.named_estimators_[name]
        joblib.dump(trained_base, f"models/{name}_model.pkl")

    print(f"Training Complete. Mean CV F1: {mean_f1:.4f} (+/- {std_f1:.4f})")