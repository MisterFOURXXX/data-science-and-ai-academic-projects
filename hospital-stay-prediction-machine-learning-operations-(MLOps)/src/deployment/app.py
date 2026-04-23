from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Hospital Stay Predictor")

model = joblib.load("models/final_stacking_model.pkl")
le = joblib.load("models/label_encoder.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
pca = joblib.load("models/pca.pkl")

class Patient(BaseModel):
    Hospital_code: int
    Hospital_type_code: str
    Hospital_region_code: str
    Available_Extra_Rooms_in_Hospital: int
    Department: str
    Ward_Type: str
    Bed_Grade: float
    Type_of_Admission: str
    Severity_of_Illness: str
    Visitors_with_Patient: int
    Age: str
    Admission_Deposit: float

@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.dict(by_alias=True)])
    df = df.rename(columns={
        "Available_Extra_Rooms_in_Hospital": "Available Extra Rooms in Hospital",
        "Bed_Grade": "Bed Grade",
        "Type_of_Admission": "Type of Admission",
        "Severity_of_Illness": "Severity of Illness"
    })
    X = preprocessor.transform(df)
    X_pca = pca.transform(X.astype(np.float32))
    pred = model.predict(X_pca)
    return {"stay_category": le.inverse_transform(pred)[0]}