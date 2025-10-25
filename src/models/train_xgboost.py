"""Train XGBoost model on IBM Telco dataset."""
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import mlflow
import mlflow.xgboost

@click.command()
@click.option('--data-path', default='data/raw/Telco-Customer-Churn.csv')
@click.option('--experiment-name', default='ibm-telco-xgboost')
@click.option('--target-accuracy', default=0.88, help='Target accuracy (88%)')
def train_model(data_path, experiment_name, target_accuracy):
    """Train XGBoost on IBM Telco Customer Churn dataset."""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Load IBM Telco data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} customers with {len(df.columns)} features")
        
        # Handle missing TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
        
        # Prepare features
        X = df.drop(['customerID', 'Churn'], axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Log dataset info
        mlflow.log_param("dataset", "IBM Telco Customer Churn")
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("churn_rate", y.mean())
        
        # Train XGBoost
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")
        print(f"Test F1: {f1:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("cv_accuracy", cv_scores.mean())
        
        # Log model if target met
        if accuracy >= target_accuracy:
            mlflow.xgboost.log_model(
                model, 
                "model",
                registered_model_name="churn_classifier_ibm"
            )
            print(f"✓ Model registered! Accuracy {accuracy:.4f} >= {target_accuracy}")
        else:
            print(f"✗ Accuracy {accuracy:.4f} < {target_accuracy}. Tuning needed.")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(importance.head(10))
        
        mlflow.log_dict(importance.to_dict(), "feature_importance.json")

if __name__ == "__main__":
    train_model()