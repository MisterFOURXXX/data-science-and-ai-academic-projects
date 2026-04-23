import numpy as np
import optuna
import mlflow
import json
import dvc.api
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from src.utils.mlflow_utils import init_mlflow

params = dvc.api.params_show()
init_mlflow()

X = np.load("data/processed/X_train.npy")
y = np.load("data/processed/y_train.npy")

def objective_lightgbm(trial):
    with mlflow.start_run(nested=True, run_name=f"LightGBM_trial_{trial.number}"):
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
        param = {
            "objective": "multiclass",
            "num_class": len(np.unique(y)),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "boosting_type": boosting_type,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
        }
        if boosting_type != "goss":
            param["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        skf = StratifiedKFold(n_splits=params["models"]["cv"], shuffle=True, random_state=params["models"]["random_state"])
        scores = [f1_score(y[val_idx], LGBMClassifier(**param, random_state=params["models"]["random_state"], n_jobs=-1, verbose=-1).fit(X[train_idx], y[train_idx]).predict(X[val_idx]), average="macro") for train_idx, val_idx in skf.split(X, y)]
        mean_f1 = np.mean(scores)
        mlflow.log_params(param)
        mlflow.log_metric("mean_macro_f1", mean_f1)
        return mean_f1

def objective_xgboost(trial):
    with mlflow.start_run(nested=True, run_name=f"XGBoost_trial_{trial.number}"):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "tree_method": trial.suggest_categorical("tree_method", ["auto", "hist"])
        }
        skf = StratifiedKFold(n_splits=params["models"]["cv"], shuffle=True, random_state=params["models"]["random_state"])
        scores = [f1_score(y[val_idx], XGBClassifier(**param, random_state=params["models"]["random_state"], n_jobs=-1, eval_metric="mlogloss", verbosity=0).fit(X[train_idx], y[train_idx]).predict(X[val_idx]), average="macro") for train_idx, val_idx in skf.split(X, y)]
        mean_f1 = np.mean(scores)
        mlflow.log_params(param)
        mlflow.log_metric("mean_macro_f1", mean_f1)
        return mean_f1

def objective_catboost(trial):
    with mlflow.start_run(nested=True, run_name=f"CatBoost_trial_{trial.number}"):
        grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
        boosting_type = trial.suggest_categorical("boosting_type", ["Plain", "Ordered"])
        if grow_policy != "SymmetricTree" and boosting_type == "Ordered":
            boosting_type = "Plain"
        param = {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "grow_policy": grow_policy,
            "boosting_type": boosting_type,
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"])
        }
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        skf = StratifiedKFold(n_splits=params["models"]["cv"], shuffle=True, random_state=params["models"]["random_state"])
        scores = [f1_score(y[val_idx], CatBoostClassifier(**param, random_seed=params["models"]["random_state"], verbose=0).fit(X[train_idx], y[train_idx], eval_set=(X[val_idx], y[val_idx]), verbose=0, early_stopping_rounds=50).predict(X[val_idx]), average="macro") for train_idx, val_idx in skf.split(X, y)]
        mean_f1 = np.mean(scores)
        mlflow.log_params(param)
        mlflow.log_metric("mean_macro_f1", mean_f1)
        return mean_f1

studies = {}
for name, obj in [("LightGBM", objective_lightgbm), ("XGBoost", objective_xgboost), ("CatBoost", objective_catboost)]:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())
    study.optimize(obj, n_trials=params["models"]["n_trials"])
    studies[name] = study.best_params

with open("models/best_weak_params.json", "w") as f:
    json.dump(studies, f)