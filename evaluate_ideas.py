import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('6a3eb196bc7a3_campus_challenge_r1_data.csv')

# Handle missing
df['f11'] = df['f11'].fillna(df['f11'].median())
for col in ['f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f21', 'f22', 'f23']:
    df[col] = df[col].fillna(0)
df['f19'] = df['f19'].fillna(df['f19'].mode()[0])
df['f20'] = df['f20'].fillna(df['f20'].mode()[0])

# Base Features
df['total_category_spend'] = df['f6'] + df['f7'] + df['f8'] + df['f9'] + df['f10']
df['spend_diversity'] = (df[['f6','f7','f8','f9','f10']] > 0).sum(axis=1)
df['digital_engagement'] = df['f12'] + df['f22'] * 2 + df['f23'] * 5
df['net_rewards_liability'] = df['f4'] - df['f21']

# Domain score V2 logic
rev_fee = (df['f20'] * 625) + ((df['f19'] - 1).clip(lower=0) * 175)
rev_interchange = (df['f6'] * 0.030 + df['f9'] * 0.030 + df['f10'] * 0.025 + df['f8'] * 0.020 + df['f7'] * 0.020)
rev_net_interest = df['f1'] * 0.15
rev_collection = df['f3'] * 10
rev_engagement = df['digital_engagement'] * 2
ead = df['f1'] + (df['f5'] / 12.0) + (df['f17'] * 0.1)
cost_credit_loss = df['f11'] * ead * 0.40
earned_points = (df['f6'] * 5 + df['f9'] * 5 + df['f10'] * 3 + df['f8'] * 2 + df['f7'] * 1)
cost_rewards = earned_points * 0.007
cost_benefits = (df['f13'] * 30 + df['f14'] * 1.0 + df['f15'] * 15 + df['f16'] * 0.8)
cost_servicing = (df['f2'] * 50 + df['f3'] * 40)
cost_future_liability = df['net_rewards_liability'].clip(lower=0) * 0.005

profitability = (rev_fee + rev_interchange + rev_net_interest + rev_engagement + rev_collection -
                 cost_credit_loss - cost_rewards - cost_benefits - cost_servicing - cost_future_liability)

df['profitability_v2'] = profitability

# Scaler
feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
                'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21']
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[feature_cols].values)

# Idea 1: PCA with different component counts
pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)
pca_3_score = np.zeros(len(df))
weights_3 = pca_3.explained_variance_ratio_
for i in range(3):
    corr = np.corrcoef(X_pca_3[:, i], df['profitability_v2'].values)[0, 1]
    sign = 1 if corr > 0 else -1
    pca_3_score += X_pca_3[:, i] * weights_3[i] * sign

print(f"PCA 3 Components Corr with Domain: {np.corrcoef(pca_3_score, df['profitability_v2'])[0,1]:.4f}")

# Idea 2: Isolation Forest for Anomaly-based Value
print("Training Isolation Forest...")
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
# Isolation forest returns negative for anomalies, positive for inliers. We want anomalies to score higher if they are high-value.
# High-value anomalies generally have high spend and high features.
iso_scores = -iso.fit_predict(X_scaled) * iso.score_samples(X_scaled) 
print(f"Iso Forest Corr with Domain: {np.corrcoef(iso_scores, df['profitability_v2'])[0,1]:.4f}")

