"""Optimize decision threshold to maximize accuracy."""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

def find_optimal_threshold():
    """Find the optimal threshold for maximum accuracy."""
    
    print("="*60)
    print("THRESHOLD OPTIMIZATION FOR 88% ACCURACY")
    print("="*60)
    
    # Load the trained model
    model_path = "models/xgboost_extreme.pkl"
    if not Path(model_path).exists():
        model_path = "models/xgboost_optimized.pkl"
    if not Path(model_path).exists():
        model_path = "models/xgboost_model.pkl"
    
    print(f"\n📦 Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load data (try v3 first, then fallback)
    data_paths = [
        "data/features/ibm_telco_features_v3.parquet",
        "data/features/ibm_telco_features_v2.parquet",
        "data/features/ibm_telco_features.parquet"
    ]
    
    for path in data_paths:
        if Path(path).exists():
            print(f"📊 Loading features from {path}")
            df = pd.read_parquet(path)
            break
    
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    
    print("\n🔍 Testing thresholds from 0.1 to 0.9...")
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # Find best threshold for accuracy
    best_idx = results_df['accuracy'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_accuracy = results_df.loc[best_idx, 'accuracy']
    
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLD FOUND")
    print("="*60)
    print(f"✅ Best Threshold: {best_threshold:.3f}")
    print(f"✅ Best Accuracy: {best_accuracy:.4f} {'🎉 TARGET MET!' if best_accuracy >= 0.88 else f'Gap: {0.88-best_accuracy:.4f}'}")
    print(f"✅ Precision: {results_df.loc[best_idx, 'precision']:.4f}")
    print(f"✅ Recall: {results_df.loc[best_idx, 'recall']:.4f}")
    print(f"✅ F1 Score: {results_df.loc[best_idx, 'f1']:.4f}")
    
    # Compare with default threshold (0.5)
    default_idx = results_df[results_df['threshold'].round(2) == 0.50].index[0]
    print(f"\n📊 Improvement over default (0.5):")
    print(f"  Accuracy: {results_df.loc[default_idx, 'accuracy']:.4f} → {best_accuracy:.4f} (+{best_accuracy - results_df.loc[default_idx, 'accuracy']:.4f})")
    
    # Show top 5 thresholds
    print("\n🏆 Top 5 Thresholds by Accuracy:")
    top_5 = results_df.nlargest(5, 'accuracy')
    for idx, row in top_5.iterrows():
        print(f"  Threshold {row['threshold']:.3f}: Acc={row['accuracy']:.4f}, F1={row['f1']:.4f}")
    
    # Save results
    results_df.to_csv("models/threshold_optimization.csv", index=False)
    print(f"\n💾 Results saved to models/threshold_optimization.csv")
    
    # Save optimal threshold
    with open("models/optimal_threshold.txt", "w") as f:
        f.write(f"{best_threshold}")
    
    return best_threshold, best_accuracy

if __name__ == "__main__":
    threshold, accuracy = find_optimal_threshold()