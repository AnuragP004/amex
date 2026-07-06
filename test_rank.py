import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('6a3eb196bc7a3_campus_challenge_r1_data.csv', nrows=50000)

df['f11'] = df['f11'].fillna(df['f11'].median())
for col in ['f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f21', 'f22', 'f23']:
    df[col] = df[col].fillna(0)
df['f19'] = df['f19'].fillna(df['f19'].mode()[0])
df['f20'] = df['f20'].fillna(df['f20'].mode()[0])

df['digital_engagement'] = df['f12'] + df['f22'] * 2 + df['f23'] * 5
df['net_rewards_liability'] = df['f4'] - df['f21']

# Domain score V2 logic
profitability = ( (df['f20'] * 625) + ((df['f19'] - 1).clip(lower=0) * 175) +
                 (df['f6'] * 0.030 + df['f9'] * 0.030 + df['f10'] * 0.025 + df['f8'] * 0.020 + df['f7'] * 0.020) +
                 df['f1'] * 0.15 + df['digital_engagement'] * 2 + df['f3'] * 10 -
                 (df['f11'] * (df['f1'] + (df['f5'] / 12.0) + (df['f17'] * 0.1)) * 0.40) -
                 ((df['f6'] * 5 + df['f9'] * 5 + df['f10'] * 3 + df['f8'] * 2 + df['f7'] * 1) * 0.007) -
                 (df['f13'] * 30 + df['f14'] * 1.0 + df['f15'] * 15 + df['f16'] * 0.8) -
                 (df['f2'] * 50 + df['f3'] * 40) - (df['net_rewards_liability'].clip(lower=0) * 0.005) )

df['profitability_v2'] = profitability

mms = MinMaxScaler()
prof_norm = mms.fit_transform(df[['profitability_v2']]).flatten()
prof_rank = df['profitability_v2'].rank(pct=True).values

print("Min-Max scaled distribution:")
print(pd.Series(prof_norm).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))

print("\nRank scaled (pct=True) distribution:")
print(pd.Series(prof_rank).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))
