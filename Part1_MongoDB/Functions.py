import pandas as pd
import numpy as np


def preprocess_data(df):
    # Rename columns to lower letters
    df = df.rename(columns=str.lower)
    
    # Removing 'id' prefix from customerID values
    if 'customerid' in df.columns:
        df['customerid'] = df['customerid'].str.replace('id', '')
    
    # Replace specific values
    df = df.replace({'no phone service': 'no', 'no internet service': 'no'}) 
    
    # Convert totalcharges to numeric before get_dummies
    if 'totalcharges' in df.columns:
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')  # Convert the string blank in totalcharges
        df['totalcharges'] = df['totalcharges'].interpolate()
    
    # Define columns to exclude from one-hot encoding
    columns_to_exclude = ['customerid']
    
    # One-hot encode categorical variables, excluding specified columns
    df_to_encode = df.drop(columns=columns_to_exclude)
    df_encoded = pd.get_dummies(df_to_encode)
    
    # Combine the encoded columns with the excluded columns
    df = pd.concat([df[columns_to_exclude], df_encoded], axis=1)
    
    # Ensure totalcharges is numeric and not affected by get_dummies
    if 'totalcharges' in df.columns:
        df['totalcharges'] = df['totalcharges'].astype(float)
    
    # Convert all data to float
    df = df.astype(float)
    
    return df

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# This is to Show what I did in the ML project to extract the ML model. The train-test was done in Project ML and downloaded the model via Pickle and joblib library

## def train_random_forest(x_train, y_train, n_estimators=9, max_depth=7):
##   clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
##   clf.fit(x_train, y_train)
##   joblib.dump(clf, 'churn_model.pkl') # I used the joblib lib and pickle to save the model (https://www.analyticsvidhya.com/blog/2021/08/quick-hacks-to-save-machine-learning-model-using-pickle-and-joblib/)
##   return clf

def load_model(model_path='churn_model.pkl'):
    clf = joblib.load(model_path)
    return clf

def predict_churn(clf, features):
    predictions = clf.predict(features)
    return predictions