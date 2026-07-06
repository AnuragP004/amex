import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('6a3eb196bc7a3_campus_challenge_r1_data.csv')

df['f11'] = df['f11'].fillna(df['f11'].median())
for col in ['f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f21', 'f22', 'f23']:
    df[col] = df[col].fillna(0)
df['f19'] = df['f19'].fillna(df['f19'].mode()[0])
df['f20'] = df['f20'].fillna(df['f20'].mode()[0])

feature_cols = ['f1', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
                'f11', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols].sample(10000, random_state=42).values)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
print(f"K-Means (k=4) counts: {np.bincount(clusters)}")
