"""Augment IBM Telco dataset to 5M rows using SMOTE + noise."""
import click
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import boto3

@click.command()
@click.option('--input', default='data/raw/Telco-Customer-Churn.csv')
@click.option('--output', required=True)
@click.option('--target-size', default=5000000)
@click.option('--method', default='smote_noise')
def augment_dataset(input, output, target_size, method):
    """Generate synthetic 5M dataset from IBM Telco 7K dataset."""
    
    print(f"Loading IBM Telco dataset from {input}")
    df = pd.read_csv(input)
    print(f"Original size: {len(df)} rows")
    
    # Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    # Prepare for SMOTE
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Apply SMOTE iteratively
    current_size = len(X)
    X_synthetic = X.copy()
    y_synthetic = y.copy()
    
    while current_size < target_size:
        print(f"Augmenting: {current_size} -> {min(current_size * 2, target_size)}")
        
        # Calculate samples needed
        n_samples = min(current_size, target_size - current_size)
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy='auto',
            k_neighbors=min(5, current_size - 1),
            random_state=42
        )
        
        X_resampled, y_resampled = smote.fit_resample(X_synthetic, y_synthetic)
        
        # Add noise to continuous features
        continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in continuous_cols:
            noise = np.random.normal(0, 0.05, size=len(X_resampled))
            X_resampled[col] = X_resampled[col] * (1 + noise)
        
        X_synthetic = X_resampled
        y_synthetic = y_resampled
        current_size = len(X_synthetic)
    
    # Trim to exact size
    X_synthetic = X_synthetic[:target_size]
    y_synthetic = y_synthetic[:target_size]
    
    # Create final dataframe
    df_synthetic = pd.DataFrame(X_synthetic, columns=X.columns)
    df_synthetic['Churn'] = y_synthetic
    df_synthetic['customerID'] = [f"SYNTH_{i:07d}" for i in range(len(df_synthetic))]
    
    # Decode categorical
    for col, encoder in encoders.items():
        df_synthetic[col] = encoder.inverse_transform(
            df_synthetic[col].astype(int)
        )
    
    df_synthetic['Churn'] = df_synthetic['Churn'].map({1: 'Yes', 0: 'No'})
    
    print(f"Generated {len(df_synthetic)} synthetic records")
    print(f"Churn rate: {(df_synthetic['Churn'] == 'Yes').mean():.2%}")
    
    # Save to S3
    if output.startswith('s3://'):
        df_synthetic.to_parquet('temp.parquet', index=False)
        s3 = boto3.client('s3')
        bucket, key = output.replace('s3://', '').split('/', 1)
        s3.upload_file('temp.parquet', bucket, key)
        print(f"Uploaded to {output}")
    else:
        df_synthetic.to_parquet(output, index=False)
        print(f"Saved to {output}")

if __name__ == "__main__":
    augment_dataset()