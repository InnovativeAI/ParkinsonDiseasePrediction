from audioop import minmax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from tqdm.notebook import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('parkinson_disease.csv')

df = df.groupby('id').mean().reset_index()
df.drop('id', axis=1, inplace = True)

columns = list(df.columns)
for col in columns:
    if col == 'class':
        continue
    
    filtered_columns = [col]
    for col1 in df.columns:
        if((col == col1) | (col == 'class')):
            continue
        
        val = df[col].corr(df[col1])
        
        if val > 0.7:
            # If the correlation between the 2
            # features is more than 0.7 remove!
            columns.remove(col1)
            continue
        else:
            filtered_columns.append(col1)
            
    # After each iteration filter out the columns
    # which are not highly correlated features.
    df = df[filtered_columns]
    
df.shape

X = df.drop('class', axis = 1)
X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k = 30)
selector.fit(X_norm, df['class'])
filtered_columns = selector.get_support()
filtered_data = X.loc[:, filtered_columns]
filtered_data['class'] = df['class']
df = filtered_data
df.shape

