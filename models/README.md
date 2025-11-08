# Models Directory

This directory contains trained models and evaluation outputs.

## Files (Generated Locally)
- `xgboost_model.pkl` - Optimized XGBoost model
- `xgboost_extreme.pkl` - Extreme tuned model
- `scaler.pkl` - Feature scaler
- `optimal_threshold.txt` - Optimal classification threshold
- `production_evaluation.json` - Model performance metrics
- `MODEL_REPORT.md` - Comprehensive model report

**Note:** Model files (.pkl) are not tracked in git due to size. Train models locally using:
```bash
python src/models/train_xgboost_extreme.py
```

## Model Performance
- AUC-ROC: 84.2%
- Balanced Accuracy: 72%
- Lift: 2.41x
- ROI: 895%
