import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

class AccidentPreprocessor:
    def __init__(self):
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='error')
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.numeric_cols = None
        self.cat_cols = None

    def temporal_embedding(self, df):
        # If your dataset has HOUR/DAY/MONTH columns use this; optional.
        for col, period in (('HOUR',24), ('DAY',31), ('MONTH',12)):
            if col in df.columns:
                df[f'{col}_SIN'] = np.sin(2 * np.pi * df[col] / period)
                df[f'{col}_COS'] = np.cos(2 * np.pi * df[col] / period)
        return df

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self.temporal_embedding(df)

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fill missing values
        df[numeric_cols] = df[numeric_cols].fillna(0)
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("missing")

        # Ordinal encode categorical columns (shift+1 to avoid zero)
        if cat_cols:
            encoded = self.ordinal_encoder.fit_transform(df[cat_cols]).astype(int)
            encoded += 1
            df[cat_cols] = encoded

        all_features = numeric_cols + cat_cols
        df[all_features] = self.scaler.fit_transform(df[all_features])

        self.fitted = True
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        return torch.FloatTensor(df[all_features].values), all_features

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit first (call fit_transform).")
        df = df.copy()
        df = self.temporal_embedding(df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        df[numeric_cols] = df[numeric_cols].fillna(0)
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("missing")
        # Try to apply ordinal encoder only on known categorical columns
        if self.cat_cols:
            # pick columns in same order
            df_cat = df[self.cat_cols].astype(str).fillna("missing")
            encoded = self.ordinal_encoder.transform(df_cat).astype(int) + 1
            df[self.cat_cols] = encoded
        all_features = self.numeric_cols + (self.cat_cols or [])
        df[all_features] = self.scaler.transform(df[all_features])
        return torch.FloatTensor(df[all_features].values)

class AccidentDataset(Dataset):
    def __init__(self, csv_path: str, preprocessor: AccidentPreprocessor=None):
        df = pd.read_csv(csv_path)
        if preprocessor is None:
            preprocessor = AccidentPreprocessor()
            self.X, self.feature_names = preprocessor.fit_transform(df)
            self.preprocessor = preprocessor
        else:
            self.preprocessor = preprocessor
            self.X = preprocessor.transform(df)
            self.feature_names = getattr(preprocessor, "numeric_cols", None) or list(range(self.X.shape[1]))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]
