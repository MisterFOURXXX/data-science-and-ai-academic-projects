import mlflow
import dagshub

def init_mlflow():
    dagshub.init(repo_owner="YOUR_DAGSHUB_USERNAME", repo_name="YOUR_DAGSHUB_REPOSITORY_NAME")                     # Revise to your DAGSHUB Information
    mlflow.set_tracking_uri(f"https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_DAGSHUB_REPOSITORY_NAME.mlflow")      # Revise to your DAGSHUB Information
    mlflow.set_experiment("YOUR_DAGSHUB_EXPERIMENT_NAME")                                                          # Revise to your DAGSHUB Information
