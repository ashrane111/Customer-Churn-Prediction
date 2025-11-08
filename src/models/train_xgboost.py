"""Train XGBoost model to achieve 88% accuracy."""
import click
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import xgboost as xgb
import mlflow
import mlflow.xgboost
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--data-path', default='data/features/ibm_telco_features.parquet')
@click.option('--experiment-name', default='ibm-telco-xgboost')
@click.option('--target-accuracy', default=0.88, type=float)
@click.option('--n-trials', default=50, help='Number of hyperparameter trials')
@click.option('--random-state', default=42)
def train_xgboost(data_path, experiment_name, target_accuracy, n_trials, random_state):
    """Train XGBoost model with hyperparameter tuning for 88% accuracy."""
    
    print("="*60)
    print(f"XGBOOST TRAINING - TARGET: {target_accuracy:.1%} ACCURACY")
    print("="*60)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:///{}".format(Path("mlruns").absolute()))
    mlflow.set_experiment(experiment_name)
    
    # Load data
    print(f"\n📊 Loading features from {data_path}")
    df = pd.read_parquet(data_path)
    
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    print(f"Shape: {X.shape}")
    print(f"Features: {list(X.columns)[:10]}...")
    print(f"Churn rate: {y.mean():.1%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Define hyperparameter search space
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.01, 0.1, 0.5, 1],
        'scale_pos_weight': [1, 2, 3]  # Handle class imbalance
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"xgboost_tuning_{n_trials}_trials"):
        
        print(f"\n🔍 Running RandomizedSearchCV with {n_trials} trials...")
        
        # Base model
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False,
            verbosity=0
        )
        
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions,
            n_iter=n_trials,
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit
        random_search.fit(X_train, y_train)
        
        # Best model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        print("\n✨ Best Parameters Found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            mlflow.log_param(param, value)
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Log metrics
        metrics = {
            'cv_best_score': random_search.best_score_,
            'test_accuracy': accuracy,
            'test_auc': auc,
            'test_f1': f1,
            'test_precision': precision,
            'test_recall': recall
        }
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        print("\n📊 Model Performance:")
        print(f"  CV Best Score: {random_search.best_score_:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f} {'✅' if accuracy >= target_accuracy else '❌'}")
        print(f"  Test AUC: {auc:.4f}")
        print(f"  Test F1: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")
        
        # Feature Importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 15 Important Features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save artifacts
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
        mlflow.log_text(importance_df.to_string(), "feature_importance.txt")
        
        # Log model
        mlflow.xgboost.log_model(
            best_model,
            "xgboost_model",
            registered_model_name="churn_classifier_ibm" if accuracy >= target_accuracy else None
        )
        
        # Save model locally
        model_path = Path("models") / "xgboost_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"\n💾 Model saved locally to {model_path}")
        
        # Check if target met
        if accuracy >= target_accuracy:
            print(f"\n🎉 SUCCESS! Target accuracy {target_accuracy:.1%} achieved!")
            mlflow.set_tag("production_ready", "true")
            
            # Log production metrics
            mlflow.log_metric("target_accuracy", target_accuracy)
            mlflow.log_metric("accuracy_surplus", accuracy - target_accuracy)
            
        else:
            gap = target_accuracy - accuracy
            print(f"\n⚠️ Target not met. Gap: {gap:.4f}")
            print("\nSuggestions to improve:")
            print("  1. Increase n_trials for more hyperparameter search")
            print("  2. Try feature engineering (polynomial features, interactions)")
            print("  3. Ensemble with other models")
            print("  4. Adjust class weights or threshold")
            
            mlflow.set_tag("production_ready", "false")
    
    return accuracy

if __name__ == "__main__":
    train_xgboost()