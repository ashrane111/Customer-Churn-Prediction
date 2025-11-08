# Data Directory Structure

## raw/
Place raw dataset files here:
- `Telco-Customer-Churn.csv` - IBM Telco dataset (download separately)

## features/
Generated feature files (created by feature engineering pipeline):
- `ibm_telco_features_v3.parquet`
- `feature_metadata.json`

## processed/
Intermediate processed data files

**Note:** Large data files (.csv, .parquet) are not tracked in git. Download/generate them locally.
