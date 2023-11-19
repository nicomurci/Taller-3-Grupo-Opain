
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to convert strings to numeric values
class StringToNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure all values are strings
        X[self.column] = X[self.column].astype(str)
        # Replace empty strings and strings with only spaces with NaN
        X[self.column] = X[self.column].apply(lambda x: pd.NA if x == "" or x.isspace() else x)
        # Convert column to numeric, coercing errors to NaN
        X[self.column] = pd.to_numeric(X[self.column], errors='coerce')
        return X
class ConvertToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['SeniorCitizen'] = X['SeniorCitizen'].astype(str)
        return X