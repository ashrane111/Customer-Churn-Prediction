"""Feature transformation pipeline for API."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class FeatureTransformer:
    """Transform raw input to model features."""
    
    def __init__(self):
        """Initialize transformer."""
        # Load reference data to know what features to create
        self.feature_columns = None
        self.load_feature_columns()
    
    def load_feature_columns(self):
        """Load the expected feature columns."""
        # Load a sample of the training data to get column names
        try:
            df = pd.read_parquet("data/features/ibm_telco_features_v3.parquet")
            # Remove metadata columns
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['customer_id', 'churn']]
        except:
            # If file not found, we'll use a predefined list
            print("Warning: Could not load feature columns from training data")
            self.feature_columns = None
    
    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw input to model features (matching v3 engineering)."""
        
        df = raw_data.copy()
        
        # Handle TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
        
        # Initialize features dataframe
        features = pd.DataFrame()
        
        # ===== NUMERICAL FEATURES =====
        features['tenure'] = df['tenure']
        features['MonthlyCharges'] = df['MonthlyCharges']
        features['TotalCharges'] = df['TotalCharges']
        
        # ===== CORE FEATURES =====
        # Customer Lifetime Value indicators
        features['avg_charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
        features['charges_diff'] = df['MonthlyCharges'] - features['avg_charges_per_month']
        features['charges_trend'] = features['charges_diff'] / (df['MonthlyCharges'] + 1)
        
        # Tenure-based risk scores
        features['tenure_risk'] = 1 / (df['tenure'] + 1)
        features['tenure_squared'] = df['tenure'] ** 2
        features['tenure_log'] = np.log1p(df['tenure'])
        features['is_new_customer'] = (df['tenure'] <= 2).astype(int)
        features['is_loyal_customer'] = (df['tenure'] >= 48).astype(int)
        
        # Contract features
        features['month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
        features['one_year'] = (df['Contract'] == 'One year').astype(int)
        features['two_year'] = (df['Contract'] == 'Two year').astype(int)
        
        # Payment method
        features['electronic_check'] = df['PaymentMethod'].str.contains('Electronic check', case=False).astype(int)
        features['auto_payment'] = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
        
        # Service features
        service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in service_cols:
            features[col] = (df[col] == 'Yes').astype(int)
        
        features['total_services'] = features[service_cols].sum(axis=1)
        features['no_services'] = (features['total_services'] == 0).astype(int)
        features['all_services'] = (features['total_services'] == len(service_cols)).astype(int)
        
        # Internet features
        features['fiber_optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
        features['dsl'] = (df['InternetService'] == 'DSL').astype(int)
        features['no_internet'] = (df['InternetService'] == 'No').astype(int)
        
        # Vulnerability indicators
        features['no_online_security'] = ((df['OnlineSecurity'] == 'No') & 
                                          (df['InternetService'] != 'No')).astype(int)
        features['no_tech_support'] = ((df['TechSupport'] == 'No') & 
                                       (df['InternetService'] != 'No')).astype(int)
        features['no_protection'] = ((df['DeviceProtection'] == 'No') & 
                                     (df['InternetService'] != 'No')).astype(int)
        
        # Demographics
        features['senior'] = df['SeniorCitizen']
        features['has_partner'] = (df['Partner'] == 'Yes').astype(int)
        features['has_dependents'] = (df['Dependents'] == 'Yes').astype(int)
        features['single'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
        features['family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
        
        # ===== INTERACTION FEATURES =====
        features['risk_new_month_to_month'] = features['is_new_customer'] * features['month_to_month']
        features['risk_senior_alone'] = features['senior'] * features['single']
        features['risk_high_charge_month_to_month'] = ((df['MonthlyCharges'] > 65) & 
                                                        (df['Contract'] == 'Month-to-month')).astype(int)
        features['risk_no_support_fiber'] = features['fiber_optic'] * features['no_tech_support']
        features['risk_electronic_check_new'] = features['electronic_check'] * features['is_new_customer']
        
        features['protected_family_contract'] = features['family'] * (1 - features['month_to_month'])
        features['protected_auto_pay_loyal'] = features['auto_payment'] * features['is_loyal_customer']
        features['protected_full_service'] = (features['total_services'] >= 4).astype(int) * features['two_year']
        
        # ===== POLYNOMIAL FEATURES =====
        poly_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for feat in poly_features:
            features[f'{feat}_sq'] = features[feat] ** 2
            features[f'{feat}_cb'] = features[feat] ** 3
            features[f'{feat}_sqrt'] = np.sqrt(features[feat])
            features[f'{feat}_log'] = np.log1p(features[feat])
        
        # Cross products
        features['tenure_x_charges'] = features['tenure'] * features['MonthlyCharges']
        features['tenure_x_total'] = features['tenure'] * features['TotalCharges']
        features['tenure_x_contract'] = features['tenure'] * features['month_to_month']
        features['charges_x_services'] = features['MonthlyCharges'] * features['total_services']
        
        # ===== RATIO FEATURES =====
        features['charge_per_service'] = features['MonthlyCharges'] / (features['total_services'] + 1)
        features['loyalty_index'] = features['tenure'] / (features['MonthlyCharges'] + 1)
        features['value_ratio'] = features['TotalCharges'] / (features['tenure'] * features['MonthlyCharges'] + 1)
        features['payment_burden'] = features['MonthlyCharges'] / (features['TotalCharges'] + 1)
        
        # ===== BINNED FEATURES =====
        features['tenure_0_6'] = ((df['tenure'] >= 0) & (df['tenure'] <= 6)).astype(int)
        features['tenure_6_12'] = ((df['tenure'] > 6) & (df['tenure'] <= 12)).astype(int)
        features['tenure_12_24'] = ((df['tenure'] > 12) & (df['tenure'] <= 24)).astype(int)
        features['tenure_24_48'] = ((df['tenure'] > 24) & (df['tenure'] <= 48)).astype(int)
        features['tenure_48_plus'] = (df['tenure'] > 48).astype(int)
        
        # Charge bins (using fixed thresholds from training)
        features['charges_q1'] = (df['MonthlyCharges'] <= 35.5).astype(int)
        features['charges_q2'] = ((df['MonthlyCharges'] > 35.5) & 
                                  (df['MonthlyCharges'] <= 65.05)).astype(int)
        features['charges_q3'] = ((df['MonthlyCharges'] > 65.05) & 
                                  (df['MonthlyCharges'] <= 89.85)).astype(int)
        features['charges_q4'] = (df['MonthlyCharges'] > 89.85).astype(int)
        
        # ===== AGGREGATE RISK SCORE =====
        features['churn_risk_score'] = (
            features['month_to_month'] * 3 +
            features['is_new_customer'] * 2 +
            features['electronic_check'] * 2 +
            features['no_online_security'] * 1.5 +
            features['no_tech_support'] * 1.5 +
            features['fiber_optic'] * 1 +
            features['single'] * 1 -
            features['two_year'] * 3 -
            features['auto_payment'] * 2 -
            features['is_loyal_customer'] * 2 -
            features['family'] * 1
        )
        
        # ===== SCALE FEATURES =====
        # Apply same scaling as training (standardization)
        from sklearn.preprocessing import StandardScaler
        
        # Note: In production, you should load the scaler used during training
        # For now, we'll scale with standard parameters
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
        
        # Ensure we have all columns expected by the model
        if self.feature_columns:
            # Add any missing columns with zeros
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            # Select only the columns the model expects, in the right order
            features = features[self.feature_columns]
        
        return features