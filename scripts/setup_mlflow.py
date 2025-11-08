"""Setup MLflow for experiment tracking."""
import mlflow
import os
from pathlib import Path

def setup_mlflow():
    """Initialize MLflow for local development."""
    # Create directories
    Path("mlruns").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Set tracking URI for local file store
    mlflow.set_tracking_uri("file:///{}".format(Path("mlruns").absolute()))
    
    # Create experiments
    experiments = {
        "ibm-telco-baseline": "Baseline models (Logistic Regression, Random Forest)",
        "ibm-telco-xgboost": "XGBoost models for 88% accuracy target",
        "ibm-telco-production": "Production-ready models"
    }
    
    for exp_name, description in experiments.items():
        try:
            exp_id = mlflow.create_experiment(
                exp_name,
                artifact_location=f"./mlruns/{exp_name}",
                tags={"description": description, "dataset": "IBM Telco"}
            )
            print(f"✅ Created experiment '{exp_name}' with ID: {exp_id}")
        except Exception as e:
            print(f"ℹ️ Experiment '{exp_name}' already exists")
    
    print("\nMLflow setup complete! Start the UI with: mlflow ui --port 5000")
    return True

if __name__ == "__main__":
    setup_mlflow()