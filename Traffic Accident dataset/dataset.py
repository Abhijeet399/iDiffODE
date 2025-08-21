import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

class AccidentPreprocessor:
    def __init__(self):
        # OrdinalEncoder with handle_unknown set so unseen categories don't crash inference
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.cat_cols = None
        self.num_cols = None

    def temporal_embedding(self, df):
        # safe: only add cols if present
        if 'HOUR' in df.columns:
            df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
            df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        if 'DAY' in df.columns:
            df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY'] / 31)
            df['DAY_COS'] = np.cos(2 * np.pi * df['DAY'] / 31)
        if 'MONTH' in df.columns:
            df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
            df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
        return df

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self.temporal_embedding(df)

        # choose numeric and categorical cols
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # fillna
        df[numeric_cols] = df[numeric_cols].fillna(0)
        for c in cat_cols:
            df[c] = df[c].astype(str).fillna("missing")

        # fit ordinal encoder on categorical columns (if any)
        if cat_cols:
            encoded = self.ordinal_encoder.fit_transform(df[cat_cols])
            # make integer and non-negative
            encoded = np.nan_to_num(encoded, nan=-1).astype(int) + 1
            df[cat_cols] = encoded

        all_features = numeric_cols + cat_cols
        # fit scaler
        df[all_features] = self.scaler.fit_transform(df[all_features])
        self.fitted = True
        self.num_cols = numeric_cols
        self.cat_cols = cat_cols
        return torch.FloatTensor(df[all_features].values)

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = df.copy()
        df = self.temporal_embedding(df)
        all_features = self.num_cols + self.cat_cols
        df[all_features] = df[all_features].fillna(0)
        for c in self.cat_cols:
            df[c] = df[c].astype(str).fillna("missing")
        # encode using existing encoder
        if self.cat_cols:
            encoded = self.ordinal_encoder.transform(df[self.cat_cols])
            encoded = np.nan_to_num(encoded, nan=-1).astype(int) + 1
            df[self.cat_cols] = encoded
        df[all_features] = self.scaler.transform(df[all_features])
        return torch.FloatTensor(df[all_features].values)


class AccidentDataset(Dataset):
    def __init__(self, csv_path, preprocessor: AccidentPreprocessor, fit=True):
        df = pd.read_csv(csv_path)
        if fit:
            self.X = preprocessor.fit_transform(df)
        else:
            self.X = preprocessor.transform(df)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]
