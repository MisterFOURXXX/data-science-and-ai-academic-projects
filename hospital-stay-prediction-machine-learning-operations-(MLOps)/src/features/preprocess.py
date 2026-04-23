import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import joblib
import dvc.api
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

params = dvc.api.params_show()
rs = params["data"]["random_state"]

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

for col in ["Age"]:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

drop_cols = ["case_id", "patientid", "City_Code_Hospital", "Ward_Facility_Code", "City_Code_Patient"]
train = train.drop(columns=drop_cols + ["Stay"])
test = test.drop(columns=drop_cols + ["Stay"])

cat_features = ["Hospital_type_code", "Hospital_region_code", "Department", "Ward_Type", "Type of Admission", "Severity of Illness", "Age"]
num_features = ["Hospital_code", "Available Extra Rooms in Hospital", "Bed Grade", "Visitors with Patient", "Admission_Deposit"]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])

preprocessor = ColumnTransformer([("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)], remainder="passthrough")
X_train = preprocessor.fit_transform(train)
X_test = preprocessor.transform(test)
joblib.dump(preprocessor, "models/preprocessor.pkl")

le = LabelEncoder()
y_train = le.fit_transform(pd.read_parquet("data/processed/train.parquet")["Stay"])
y_test = le.transform(pd.read_parquet("data/processed/test.parquet")["Stay"])
joblib.dump(le, "models/label_encoder.pkl")

sm = SMOTE(random_state=rs)
X_res, y_res = sm.fit_resample(X_train, y_train)

pca = PCA(n_components=params["preprocess"]["pca_variance"], random_state=rs)
X_train_pca = pca.fit_transform(X_res.astype(np.float32))
X_test_pca = pca.transform(X_test.astype(np.float32))
joblib.dump(pca, "models/pca.pkl")

np.save("data/processed/X_train.npy", X_train_pca)
np.save("data/processed/y_train.npy", y_res)
np.save("data/processed/X_test.npy", X_test_pca)
np.save("data/processed/y_test.npy", y_test)