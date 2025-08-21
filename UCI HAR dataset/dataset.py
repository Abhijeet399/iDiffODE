import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split

def load_uci_har(data_dir: str):
    """
      data_dir/
        train/X_train.txt
        test/X_test.txt
    """
    def read_split(split):
        X_path = os.path.join(data_dir, split, f"X_{split}.txt")
        if not os.path.exists(X_path):
            # try alternate filenames historically used
            alt = os.path.join(data_dir, split, f"X_{split}.txt")
            raise FileNotFoundError(f"Could not find {X_path}")
        return pd.read_csv(X_path, delim_whitespace=True, header=None)
    df_train = read_split("train")
    df_test = read_split("test")
    return pd.concat([df_train, df_test], ignore_index=True)


class HARPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, df: pd.DataFrame):
        arr = df.fillna(0).values.astype(float)
        arr_scaled = self.scaler.fit_transform(arr)
        return torch.FloatTensor(arr_scaled)


class HARRawDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor):
        self.X = data_tensor

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx]


def get_dataloaders(data_dir: str, batch_size: int = 16, val_split: float = 0.2, num_workers: int = 0):
    df = load_uci_har(data_dir)
    pre = HARPreprocessor()
    tensor_data = pre.preprocess(df)
    dataset = HARRawDataset(tensor_data)

    train_size = int((1.0 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, tensor_data.shape[1]
