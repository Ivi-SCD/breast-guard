import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

ds = pd.read_csv('./data/raw/breast_cancer_analyzed.csv')
ds['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)

manual_features = ds.iloc[:, 1:].values
target = ds.iloc[:, 0].values

standard_features = StandardScaler().fit_transform(manual_features)
minmax_features = MinMaxScaler().fit_transform(manual_features)

with open('./data/processed/data.pkl', 'wb') as f:
    pickle.dump({
        'target': target,
        'manual_features': manual_features,
        'standard_features': standard_features,
        'minmax_features': minmax_features
    }, f)