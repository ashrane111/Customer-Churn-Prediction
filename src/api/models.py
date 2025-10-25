"""Pydantic models for IBM Telco Customer Churn dataset."""
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, validator

class IBMTelcoFeatures(BaseModel):
    """Input features from IBM Telco dataset (7,043 customers)."""
    
    # Customer info
    customerID: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., regex="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., regex="^(Yes|No)$")
    Dependents: str = Field(..., regex="^(Yes|No)$")
    
    # Account info
    tenure: int = Field(..., ge=0, le=100, description="Months with company")
    Contract: str = Field(..., regex="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., regex="^(Yes|No)$")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, etc")
    MonthlyCharges: float = Field(..., ge=0, le=200)
    TotalCharges: float = Field(..., ge=0)
    
    # Services
    PhoneService: str = Field(..., regex="^(Yes|No)$")
    MultipleLines: str = Field(..., regex="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., regex="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(..., regex="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., regex="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., regex="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., regex="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., regex="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., regex="^(Yes|No|No internet service)$")
    
    @validator('TotalCharges')
    def validate_total_charges(cls, v, values):
        """Handle missing TotalCharges for new customers."""
        if v == 0 and 'tenure' in values and values['tenure'] == 0:
            return values.get('MonthlyCharges', 0)
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 358.2,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    feature_importance: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    environment: str