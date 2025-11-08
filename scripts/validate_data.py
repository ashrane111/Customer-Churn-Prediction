"""Validate data quality for IBM Telco dataset."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def validate_ibm_telco_data(data_path):
    """Validate IBM Telco data quality."""
    
    print("Running data validation checks...")
    errors = []
    warnings = []
    
    # Load data
    df = pd.read_csv(data_path)
    
    # 1. Check shape
    if df.shape != (7043, 21):
        warnings.append(f"Unexpected shape: {df.shape}. Expected (7043, 21)")
    
    # 2. Check columns
    expected_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # 3. Check data types
    if df['tenure'].dtype not in ['int64', 'float64']:
        errors.append(f"tenure should be numeric, got {df['tenure'].dtype}")
    
    if df['MonthlyCharges'].dtype not in ['float64']:
        errors.append(f"MonthlyCharges should be float, got {df['MonthlyCharges'].dtype}")
    
    # 4. Check value ranges
    if df['tenure'].min() < 0 or df['tenure'].max() > 100:
        errors.append(f"tenure out of range: [{df['tenure'].min()}, {df['tenure'].max()}]")
    
    if df['MonthlyCharges'].min() < 0 or df['MonthlyCharges'].max() > 200:
        warnings.append(f"MonthlyCharges unusual range: [{df['MonthlyCharges'].min()}, {df['MonthlyCharges'].max()}]")
    
    # 5. Check categorical values
    if not set(df['Churn'].unique()).issubset({'Yes', 'No'}):
        errors.append(f"Invalid Churn values: {df['Churn'].unique()}")
    
    if not set(df['Contract'].unique()).issubset({'Month-to-month', 'One year', 'Two year'}):
        errors.append(f"Invalid Contract values: {df['Contract'].unique()}")
    
    # 6. Check for duplicates
    duplicates = df['customerID'].duplicated().sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate customerIDs")
    
    # 7. Check TotalCharges special case
    if df['TotalCharges'].dtype == 'object':
        empty_total = (df['TotalCharges'] == ' ').sum()
        if empty_total > 0:
            warnings.append(f"Found {empty_total} empty TotalCharges (expected for new customers)")
    
    # Print results
    print("\n" + "=" * 60)
    print("DATA VALIDATION RESULTS")
    print("=" * 60)
    
    if not errors and not warnings:
        print("✅ All validation checks passed!")
    else:
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"  - {warning}")
    
    print("\nDataset Summary:")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns)}")
    print(f"  - Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    
    return len(errors) == 0

if __name__ == "__main__":
    data_path = 'data/raw/Telco-Customer-Churn.csv'
    if not Path(data_path).exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)
    
    is_valid = validate_ibm_telco_data(data_path)
    sys.exit(0 if is_valid else 1)