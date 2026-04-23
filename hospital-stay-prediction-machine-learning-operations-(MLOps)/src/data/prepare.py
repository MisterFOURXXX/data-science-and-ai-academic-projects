import pandas as pd
from sklearn.model_selection import train_test_split
import dvc.api
import os

os.makedirs("data/processed", exist_ok=True)

params = dvc.api.params_show()
df = pd.read_csv("data/raw/train_data.csv")
df = df.sample(n=params["data"]["sample_size"], random_state=params["data"]["random_state"])

train, test = train_test_split(df, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"], stratify=df["Stay"])

train.to_parquet("data/processed/train.parquet", index=False)
test.to_parquet("data/processed/test.parquet", index=False)