"""Test the production API."""
import requests
import json

# Test data
test_customer = {
    "customerID": "test-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
}

# Test endpoints
base_url = "http://localhost:8000"

# 1. Test root
response = requests.get(f"{base_url}/")
print("Root:", response.json())

# 2. Test health
response = requests.get(f"{base_url}/health")
print("Health:", response.json())

# 3. Test prediction
response = requests.post(f"{base_url}/predict", json=test_customer)
print("Prediction:", json.dumps(response.json(), indent=2))

# 4. Test model info
response = requests.get(f"{base_url}/model/info")
print("Model Info:", json.dumps(response.json(), indent=2))