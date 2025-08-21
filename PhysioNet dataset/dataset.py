# dataset.py
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

class PhysioNetPreprocessor:
    def __init__(self, scaler=None):
        self.scaler = scaler or MinMaxScaler()

    def load_and_aggregate(self, data_dir, pattern="*.csv"):
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    def preprocess(self, df, numeric_only=True, fit=True):
        df = df.fillna(0)
        if numeric_only:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            data = df[numeric_cols].values
        else:
            data = df.values
        if fit:
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = self.scaler.transform(data)
        return torch.FloatTensor(data_scaled), self.scaler

class PhysioNetDataset(Dataset):
    def __init__(self, data_tensor):
        assert isinstance(data_tensor, torch.Tensor)
        self.X = data_tensor

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx]

def get_dataloaders(data_dir,
                    batch_size=8,
                    val_split=0.2,
                    num_workers=0,
                    random_seed=42):
    pre = PhysioNetPreprocessor()
    df = pre.load_and_aggregate(data_dir)
    X, scaler = pre.preprocess(df)
    dataset = PhysioNetDataset(X)

    n_total = len(dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    torch.manual_seed(random_seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    input_dim = X.shape[1]
    return train_loader, val_loader, input_dim, scaler
