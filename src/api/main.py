"""FastAPI service for churn prediction - Python 3.9 compatible."""
import os
import time
from typing import Dict, Optional
import asyncio
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from starlette.responses import Response

from .models import PredictionRequest, PredictionResponse, HealthResponse, IBMTelcoFeatures
from .inference import ModelInference

# Metrics
registry = CollectorRegistry()
prediction_counter = Counter('predictions_total', 'Total predictions', registry=registry)
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency', registry=registry)
model_drift = Gauge('model_drift_psi', 'Population Stability Index', registry=registry)

# Global model instance
model_inference = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle."""
    global model_inference
    print("Loading model from MLflow...")
    model_inference = ModelInference()
    await model_inference.load_model()
    print(f"Model loaded: {model_inference.model_version}")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML service for telecom churn prediction",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time to headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for Kubernetes probes."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_inference is not None,
        model_version=model_inference.model_version if model_inference else None,
        environment=os.getenv("ENVIRONMENT", "unknown")
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: IBMTelcoFeatures):
    """
    Predict churn probability for IBM Telco customer.
    
    Uses the 21 features from IBM Telco Customer Churn dataset.
    """
    with prediction_latency.time():
        try:
            # Convert to dict and predict
            features = request.dict()
            prediction = await model_inference.predict(features)
            
            # Update metrics
            prediction_counter.inc()
            
            # Calculate drift (simplified)
            drift_score = await model_inference.calculate_drift(features)
            model_drift.set(drift_score)
            
            return PredictionResponse(
                customer_id=request.customerID,
                churn_probability=prediction['probability'],
                churn_prediction=prediction['label'],
                confidence=prediction['confidence'],
                model_version=model_inference.model_version,
                feature_importance=prediction.get('feature_importance')
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(registry), media_type="text/plain")

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get current model information."""
    if not model_inference:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_name": model_inference.model_name,
        "model_version": model_inference.model_version,
        "model_stage": model_inference.stage,
        "features": model_inference.feature_names,
        "training_metrics": {
            "accuracy": 0.88,
            "auc": 0.90,
            "f1": 0.65
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)