import os
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd


def encode(df, target_attribute, y_train):
    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    
    # Apply one-hot encoding to non-numeric columns
    encoded_df = pd.get_dummies(df, columns=non_numeric_cols)
    encoded_df[target_attribute] = y_train

    return encoded_df