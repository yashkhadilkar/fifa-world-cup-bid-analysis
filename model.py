import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

# ----------------------------------------------------------
# 1. Load training features (host countries only)
# ----------------------------------------------------------
train = pd.read_parquet('gs://msba405-team-1-data/processed/training_features/')
print(f"Training data: {train.shape}")

indicator_cols = [c for c in train.columns if c not in ('host_iso3', 'tournament_year')]

# ----------------------------------------------------------
# 2. Impute missing values (median) and scale
# ----------------------------------------------------------
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train = train[indicator_cols].values
X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

print(f"Features after impute/scale: {X_train_scaled.shape}")
print(f"Any NaNs remaining: {np.isnan(X_train_scaled).any()}")

# ----------------------------------------------------------
# 3. Train Isolation Forest
# ----------------------------------------------------------
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.1,   # expect ~10% of hosts to look "unusual"
    max_samples='auto',
    random_state=42
)

iso_forest.fit(X_train_scaled)

# Score the training set to see how hosts score
train['anomaly_score'] = iso_forest.decision_function(X_train_scaled)
train['anomaly_label'] = iso_forest.predict(X_train_scaled)  # 1 = normal, -1 = anomaly

print(f"\nHost country anomaly scores:")
print(train[['host_iso3', 'tournament_year', 'anomaly_score', 'anomaly_label']]
      .sort_values('anomaly_score')
      .to_string())

# ----------------------------------------------------------
# 4. Score ALL countries (prediction step)
# ----------------------------------------------------------
# Load full indicator data, build latest-available features for every country
wdi = pd.read_parquet('gs://msba405-team-1-data/raw/wdi/wdi_data.parquet')
imf = pd.read_parquet('gs://msba405-team-1-data/raw/imf/imf_data.parquet')

wdi['source'] = 'WDI'
imf['source'] = 'IMF'
all_indicators = pd.concat([wdi, imf], ignore_index=True)

# For each country, use the most recent 6 years of data as features
# (simulates "if they bid for an upcoming tournament")
latest_year = all_indicators['year'].max()
window_start = latest_year - 5  # 6-year window

recent = all_indicators[all_indicators['year'] >= window_start]
country_features_long = (recent
    .groupby(['iso3', 'indicator_code'])['value']
    .mean()
    .reset_index()
)

# Pivot to wide
country_features_wide = country_features_long.pivot(
    index='iso3', columns='indicator_code', values='value'
).reset_index()

# Align columns to training set
missing_cols = set(indicator_cols) - set(country_features_wide.columns)
for col in missing_cols:
    country_features_wide[col] = np.nan

X_all = country_features_wide[indicator_cols].values
X_all_imputed = imputer.transform(X_all)
X_all_scaled = scaler.transform(X_all_imputed)

# Score all countries
country_features_wide['hosting_readiness_score'] = iso_forest.decision_function(X_all_scaled)
country_features_wide['anomaly_label'] = iso_forest.predict(X_all_scaled)

# ----------------------------------------------------------
# 5. Save predictions
# ----------------------------------------------------------
predictions = country_features_wide[['iso3', 'hosting_readiness_score', 'anomaly_label']].copy()
predictions = predictions.sort_values('hosting_readiness_score', ascending=False)

print(f"\nTop 20 countries by hosting readiness:")
print(predictions.head(20).to_string())

print(f"\nBottom 10 countries (least ready):")
print(predictions.tail(10).to_string())

# Save to GCS
predictions.to_parquet('gs://msba405-team-1-data/processed/predictions/country_scores.parquet', index=False)
print(f"\nPredictions saved: {predictions.shape[0]} countries scored")

# Save model artifacts
import joblib
joblib.dump(iso_forest, 'iso_forest_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')

!gsutil cp iso_forest_model.pkl gs://msba405-team-1-data/models/iso_forest_model.pkl
!gsutil cp imputer.pkl gs://msba405-team-1-data/models/imputer.pkl
!gsutil cp scaler.pkl gs://msba405-team-1-data/models/scaler.pkl

print("Model artifacts saved to gs://msba405-team-1-data/models/")