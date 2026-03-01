import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pipeline = None
    
    def fill_missing_values(self, df):
        """Fill missing values with median"""
        df['Credit_Score'] = df['Credit_Score'].fillna(df['Credit_Score'].median())
        df['Income_Missing_Flag'] = df['Income'].isnull().astype(int)
        df['Income'] = df['Income'].fillna(df['Income'].median())
        df['Loan_Balance_Missing_Flag'] = df['Loan_Balance'].isnull().astype(int)
        df['Loan_Balance'] = df['Loan_Balance'].fillna(df['Loan_Balance'].median())
        return df
    
    def encode_categorical(self, df):
        """Encode categorical payment columns"""
        mapping = {'Late': 0, 'Missed': 1, 'On-time': 2}
        cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
        df[cols] = df[cols].replace(mapping)
        df = df.astype({col: 'int' for col in cols})
        return df
    
    def engineer_features(self, df):
        """Create new features"""
        df['Missed_Payments_6_months'] = df[['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']].eq(1).sum(axis=1)
        df['Late_Payments_6_months'] = df[['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']].eq(0).sum(axis=1)
        df['Trend_Score'] = df['Month_1']*5 + df['Month_4']*1 + df['Month_6']*3
        df['Stress_index'] = df['Credit_Utilization'] * df['Debt_to_Income_Ratio']
        df['Loan_Balance_to_income'] = df['Loan_Balance'] / df['Income']
        df['Missed_to_Tenure_Ratio'] = df['Missed_Payments'] / (df['Account_Tenure'] + 1)
        df['High_Utilization'] = (df['Credit_Utilization'] > 0.8).astype(int)
        df['High_DTI'] = (df['Debt_to_Income_Ratio'] > 0.4).astype(int)
        df['Low_Credit_Score'] = (df['Credit_Score'] < 600).astype(int)
        return df
    
    def preprocess(self, df, fit=False):
        """Execute full preprocessing pipeline"""
        df = df.copy()
        df = self.fill_missing_values(df)
        df = self.encode_categorical(df)
        df = self.engineer_features(df)
        return df