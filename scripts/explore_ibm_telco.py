"""Explore and analyze IBM Telco Customer Churn dataset."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def explore_dataset(data_path='data/raw/Telco-Customer-Churn.csv'):
    """Comprehensive exploration of IBM Telco dataset."""
    
    print("=" * 80)
    print("IBM TELCO CUSTOMER CHURN DATASET EXPLORATION")
    print("=" * 80)
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # 1. Basic Information
    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. Column Information
    print("\n2. COLUMN INFORMATION")
    print("-" * 40)
    print(df.info())
    
    # 3. Target Variable Distribution
    print("\n3. TARGET VARIABLE (CHURN) DISTRIBUTION")
    print("-" * 40)
    churn_dist = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    print(f"No Churn: {churn_dist['No']} ({churn_pct['No']:.1f}%)")
    print(f"Churn: {churn_dist['Yes']} ({churn_pct['Yes']:.1f}%)")
    print(f"Class Imbalance Ratio: {churn_dist['No']/churn_dist['Yes']:.2f}:1")
    
    # 4. Missing Values
    print("\n4. MISSING VALUES")
    print("-" * 40)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values detected!")
    else:
        print(missing[missing > 0])
    
    # Special case: TotalCharges has empty strings
    total_charges_missing = (df['TotalCharges'] == ' ').sum()
    if total_charges_missing > 0:
        print(f"TotalCharges has {total_charges_missing} empty string values (new customers)")
    
    # 5. Feature Types
    print("\n5. FEATURE TYPES")
    print("-" * 40)
    
    # Identify feature types
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                      'PhoneService', 'PaperlessBilling']
    categorical_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    print(f"Numerical Features ({len(numerical_features)}): {', '.join(numerical_features)}")
    print(f"\nBinary Features ({len(binary_features)}): {', '.join(binary_features)}")
    print(f"\nCategorical Features ({len(categorical_features)}): {', '.join(categorical_features)}")
    
    # 6. Numerical Features Statistics
    print("\n6. NUMERICAL FEATURES STATISTICS")
    print("-" * 40)
    
    # Fix TotalCharges
    df_clean = df.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'].fillna(df_clean['MonthlyCharges'], inplace=True)
    
    print(df_clean[numerical_features].describe())
    
    # 7. Categorical Features Distribution
    print("\n7. CATEGORICAL FEATURES UNIQUE VALUES")
    print("-" * 40)
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['customerID', 'Churn']:
            unique_vals = df[col].unique()
            print(f"{col}: {unique_vals}")
    
    # 8. Correlations with Churn
    print("\n8. TOP FEATURES CORRELATED WITH CHURN")
    print("-" * 40)
    
    # Convert categorical to numerical for correlation
    df_encoded = df_clean.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    
    correlations = df_encoded.corr()['Churn'].sort_values(ascending=False)
    print("Positive correlations (increases churn):")
    print(correlations[correlations > 0.1].head(10))
    print("\nNegative correlations (decreases churn):")
    print(correlations[correlations < -0.1].head(10))
    
    # 9. Business Insights
    print("\n9. KEY BUSINESS INSIGHTS")
    print("-" * 40)
    
    # Average tenure by churn
    avg_tenure_churn = df_clean[df_clean['Churn'] == 'Yes']['tenure'].mean()
    avg_tenure_no_churn = df_clean[df_clean['Churn'] == 'No']['tenure'].mean()
    print(f"Average tenure - Churned: {avg_tenure_churn:.1f} months")
    print(f"Average tenure - Retained: {avg_tenure_no_churn:.1f} months")
    
    # Contract type impact
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean() * 100)
    print(f"\nChurn rate by contract type:")
    for contract, rate in contract_churn.items():
        print(f"  {contract}: {rate:.1f}%")
    
    # Internet service impact
    internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x=='Yes').mean() * 100)
    print(f"\nChurn rate by internet service:")
    for service, rate in internet_churn.items():
        print(f"  {service}: {rate:.1f}%")
    
    # 10. Feature Engineering Opportunities
    print("\n10. FEATURE ENGINEERING OPPORTUNITIES")
    print("-" * 40)
    print("✓ tenure_group: Categorize tenure into bins (new/medium/long-term)")
    print("✓ avg_monthly_spend: TotalCharges / tenure")
    print("✓ has_streaming: Combine StreamingTV and StreamingMovies")
    print("✓ has_security: Combine OnlineSecurity and OnlineBackup")
    print("✓ contract_monthly: Binary flag for month-to-month")
    print("✓ payment_electronic: Binary flag for electronic payment")
    print("✓ services_count: Total number of services subscribed")
    
    # Save summary to JSON
    summary = {
        'dataset': 'IBM Telco Customer Churn',
        'n_samples': int(df.shape[0]),
        'n_features': int(df.shape[1]),
        'churn_rate': float(churn_pct['Yes']),
        'numerical_features': numerical_features,
        'binary_features': binary_features,
        'categorical_features': categorical_features,
        'missing_values': int(missing.sum() + total_charges_missing),
        'class_imbalance_ratio': float(churn_dist['No']/churn_dist['Yes'])
    }
    
    with open('data/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Exploration complete! Summary saved to data/dataset_summary.json")
    print("=" * 80)
    
    return df_clean

if __name__ == "__main__":
    df = explore_dataset()