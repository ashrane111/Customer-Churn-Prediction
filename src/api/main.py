"""Production FastAPI service with proper feature engineering."""
import os
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .models import IBMTelcoFeatures, PredictionResponse, HealthResponse
from .feature_transformer import FeatureTransformer

# Global variables
model = None
feature_transformer = None
optimal_threshold = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and feature transformer on startup."""
    global model, feature_transformer, optimal_threshold
    
    print("Loading production model...")
    model = joblib.load("models/xgboost_extreme.pkl")
    
    print("Loading feature transformer...")
    feature_transformer = FeatureTransformer()
    
    with open("models/optimal_threshold.txt", "r") as f:
        optimal_threshold = float(f.read())
    
    print(f"Model loaded. Threshold: {optimal_threshold:.3f}")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production API with 84.2% AUC-ROC and 2.41x Lift",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "api": "Customer Churn Prediction",
        "model_performance": {
            "AUC-ROC": 0.842,
            "Balanced_Accuracy": 0.707,
            "Lift": 2.41,
            "ROI": "895%"
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="xgboost_extreme_v1",
        environment=os.getenv("ENVIRONMENT", "production"),
        threshold=optimal_threshold
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IBMTelcoFeatures):
    """
    Predict churn for a single customer.
    
    Returns:
    - churn_probability: Raw probability (0-1)
    - churn_prediction: Binary prediction based on optimal threshold
    - risk_level: HIGH/MEDIUM/LOW
    - retention_action: Recommended action
    """
    if model is None or feature_transformer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert input to dataframe
        input_df = pd.DataFrame([features.dict()])
        
        # Apply feature engineering (same as training)
        processed_features = feature_transformer.transform(input_df)
        
        # Get probability
        probability = model.predict_proba(processed_features)[0, 1]
        
        # Business logic
        prediction = int(probability >= optimal_threshold)
        
        # Risk levels
        if probability >= 0.75:
            risk_level = "HIGH"
            action = "Immediate retention offer recommended ($50-100 credit)"
        elif probability >= 0.5:
            risk_level = "MEDIUM"
            action = "Monitor closely, consider retention offer ($25 credit)"
        else:
            risk_level = "LOW"
            action = "No immediate action needed"
        
        # Calculate expected value
        retention_cost = 50
        customer_value = 1000
        
        if prediction:
            expected_value = probability * (customer_value - retention_cost) - (1-probability) * retention_cost
        else:
            expected_value = 0
        
        return PredictionResponse(
            customer_id=features.customerID,
            churn_probability=float(probability),
            churn_prediction=prediction,
            risk_level=risk_level,
            retention_action=action,
            expected_value=float(expected_value),
            confidence=float(abs(probability - 0.5) * 2),
            model_version="xgboost_extreme_v1"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information and performance metrics."""
    return {
        "model_type": "XGBoost",
        "version": "xgboost_extreme_v1",
        "threshold": optimal_threshold,
        "performance_metrics": {
            "AUC-ROC": 0.8416,
            "Balanced_Accuracy": 0.7066,
            "F1_Score": 0.5728,
            "Precision": 0.6402,
            "Recall": 0.5182,
            "Matthews_Correlation": 0.4441,
            "Lift": 2.41,
            "ROI": "895%"
        },
        "business_impact": {
            "description": "2.41x better than random selection",
            "retention_cost": "$50 per customer",
            "expected_roi": "895% on retention campaigns"
        },
        "features_used": 100,
        "training_date": "2024",
        "dataset": "IBM Telco Customer Churn (7,043 samples)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)