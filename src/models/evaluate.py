"""Evaluate trained model performance."""
import click
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from pathlib import Path
import json

@click.command()
@click.option('--model-name', default='churn_classifier_ibm', help='Registered model name')
@click.option('--test-data', default='data/features/ibm_telco_features.parquet')
@click.option('--stage', default='None', help='Model stage (None/Staging/Production)')
def evaluate_model(model_name, test_data, stage):
    """Evaluate registered model on test data."""
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:///{}".format(Path("mlruns").absolute()))
    client = MlflowClient()
    
    try:
        # Get latest model version
        if stage == 'None':
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"❌ No model found with name '{model_name}'")
                print("Available models:")
                for rm in client.list_registered_models():
                    print(f"  - {rm.name}")
                return
            
            latest_version = max(versions, key=lambda x: int(x.version))
        else:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                print(f"❌ No model found in stage '{stage}'")
                return
            latest_version = versions[0]
        
        print(f"\n📦 Loading model: {model_name} v{latest_version.version}")
        
        # Load model
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Load test data
        print(f"📊 Loading test data from {test_data}")
        df = pd.read_parquet(test_data)
        
        X_test = df.drop(['customer_id', 'churn'], axis=1)
        y_test = df['churn']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For probability scores (if available)
        try:
            # Try to get the underlying sklearn model
            if hasattr(model._model_impl, 'predict_proba'):
                y_pred_proba = model._model_impl.predict_proba(X_test)[:, 1]
            else:
                # XGBoost through MLflow
                import xgboost as xgb
                if isinstance(model._model_impl, xgb.XGBClassifier):
                    y_pred_proba = model._model_impl.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = y_pred  # Fallback
        except:
            y_pred_proba = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        f1 = f1_score(y_test, y_pred)
        
        print("\n📈 Model Performance:")
        print(f"  Accuracy: {accuracy:.4f} {'✅' if accuracy >= 0.88 else ''}")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Business metrics
        tn = ((y_test == 0) & (y_pred == 0)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        fn = ((y_test == 1) & (y_pred == 0)).sum()
        tp = ((y_test == 1) & (y_pred == 1)).sum()
        
        print("\n💼 Business Impact:")
        print(f"  Correctly identified churners: {tp}/{tp+fn} ({tp/(tp+fn):.1%})")
        print(f"  False alarms: {fp}/{fp+tn} ({fp/(fp+tn):.1%})")
        print(f"  Missed churners: {fn}/{tp+fn} ({fn/(tp+fn):.1%})")
        
        # Cost-benefit analysis (example values)
        retention_cost = 50  # Cost to retain a customer
        churn_cost = 500    # Cost of losing a customer
        
        savings = tp * (churn_cost - retention_cost)  # Saved churners
        waste = fp * retention_cost  # False positives
        loss = fn * churn_cost  # Missed churners
        net_benefit = savings - waste - loss
        
        print(f"\n💰 Cost-Benefit Analysis (example):")
        print(f"  Retention cost: ${retention_cost}")
        print(f"  Churn cost: ${churn_cost}")
        print(f"  Potential savings: ${savings:,}")
        print(f"  Wasted on false positives: ${waste:,}")
        print(f"  Loss from missed churners: ${loss:,}")
        print(f"  Net benefit: ${net_benefit:,}")
        
        # Save evaluation results
        results = {
            'model_name': model_name,
            'model_version': int(latest_version.version),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'f1': float(f1),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            },
            'business_metrics': {
                'retention_rate': float(tp/(tp+fn)),
                'false_positive_rate': float(fp/(fp+tn)),
                'net_benefit': float(net_benefit)
            }
        }
        
        eval_path = Path("models") / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Evaluation results saved to {eval_path}")
        
        # Update model stage if performance is good
        if accuracy >= 0.88 and stage == 'None':
            print(f"\n🚀 Model meets production criteria (88% accuracy)")
            response = input("Promote to Staging? (y/n): ")
            if response.lower() == 'y':
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Staging"
                )
                print("✅ Model promoted to Staging!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model()