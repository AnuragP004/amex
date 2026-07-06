import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('6a3eb196bc7a3_campus_challenge_r1_data.csv', nrows=10000)
print(df.columns)
