#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import numpy as np
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import umap
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


# In[2]:


class AccidentPreprocessor:
    def __init__(self):
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='error')
        self.scaler = MinMaxScaler()

    def temporal_embedding(self, df):
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY'] / 31)
        df['DAY_COS'] = np.cos(2 * np.pi * df['DAY'] / 31)
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
        return df

    def preprocess(self, df):
        df = df.copy()

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        df[numeric_cols] = df[numeric_cols].fillna(0)

        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("missing")

        if cat_cols:
            encoded = self.ordinal_encoder.fit_transform(df[cat_cols]).astype(int)
            encoded += 1  # shift all encoded values to be positive (0 → 1, 1 → 2, etc.)
            df[cat_cols] = encoded

        all_features = numeric_cols + cat_cols
        df[all_features] = self.scaler.fit_transform(df[all_features])

        return torch.FloatTensor(df.values)


# In[3]:


class AccidentDataset(Dataset):
    def __init__(self, csv_path, preprocessor):
        df = pd.read_csv(csv_path)
        self.X = preprocessor.preprocess(df)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx]


# In[4]:


class TemporalODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, t, x):
        t_tensor = torch.ones_like(x[:, :1]) * t
        x_with_time = torch.cat([x, t_tensor], dim=1)
        return self.net(x_with_time)

class NeuralODEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ode_func = TemporalODEFunc(hidden_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x, time_points):
        # x: [batch, features]
        h = self.input_proj(x)
        h = odeint(self.ode_func, h, time_points, method='dopri5')[-1]
        latent = self.latent_proj(h)
        return latent


# In[5]:


class AccidentDiffusion(nn.Module):
    def __init__(self, latent_dim, hidden_dim, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        # Register as buffers so they move with the model
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1).to(x.device)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1).to(x.device)
        epsilon = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * epsilon, epsilon
    
    def forward(self, x):
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,)).to(x.device)
        noisy_x, noise = self.add_noise(x, t)
        t_norm = t.float() / self.timesteps
        predicted_noise = self.noise_predictor(torch.cat([noisy_x, t_norm.unsqueeze(1)], dim=1))
        return noisy_x, noise, predicted_noise
    
    def sample(self, x, steps=100):
        for t in reversed(range(steps)):
            t_tensor = torch.full((x.size(0), 1), t/steps).to(x.device)
            noise_pred = self.noise_predictor(torch.cat([x, t_tensor], dim=1))
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) / torch.sqrt(alpha_t)
            x += torch.sqrt(beta_t) * noise
        return x


# In[6]:


class InverseResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, y, n_iter=10):
        # Fixed-point iteration for inversion: x_{i+1} = y - g(x_i)
        x = y
        for _ in range(n_iter):
            x = y - self.g(x)
        return x

class InverseResNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_blocks=4):
        super().__init__()
        self.initial = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([InverseResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.final = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return x


# In[7]:


class AccidentPredictionPipeline(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = NeuralODEEncoder(input_dim, hidden_dim, latent_dim)
        self.diffusion = AccidentDiffusion(latent_dim, hidden_dim)
        self.inverse_resnet = InverseResNet(latent_dim, hidden_dim, input_dim)
        
    def forward(self, x, time_points):
        lode = self.encoder(x, time_points)
        noisy_lode, noise, pred_noise = self.diffusion(lode)
        reconstructed_lode = self.diffusion.sample(noisy_lode)
        reconstructed_input = self.inverse_resnet(reconstructed_lode)
        return {
            'lode': lode,
            'noisy_lode': noisy_lode,
            'reconstructed_lode': reconstructed_lode,
            'reconstructed_input': reconstructed_input,
            'noise': noise,
            'predicted_noise': pred_noise
        }


# In[8]:


def composite_loss(outputs, original):
    lode_loss = nn.MSELoss()(outputs['lode'], outputs['reconstructed_lode'])
    recon_loss = nn.MSELoss()(original, outputs['reconstructed_input'])
    diffusion_loss = nn.MSELoss()(outputs['noise'], outputs['predicted_noise'])
    return lode_loss + recon_loss + diffusion_loss


# In[9]:


def save_model_outputs_to_csv(model, data_loader, time_points, device, save_path, epoch):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x, time_points)
            
            original_input = batch_x.detach().cpu()
            reconstructed_input = outputs["reconstructed_input"].detach().cpu()
            lode = outputs["lode"].detach().cpu()
            reconstructed_lode = outputs["reconstructed_lode"].detach().cpu()

            for i in range(original_input.size(0)):
                rows.append(["Original Input (Epoch {})".format(epoch + 1)] + original_input[i].tolist())
                rows.append(["Reconstructed Input"] + reconstructed_input[i].tolist())
                rows.append(["Latent Vector (lode)"] + lode[i].tolist())
                rows.append(["Reconstructed Latent"] + reconstructed_lode[i].tolist())
                rows.append([])  # Empty row for readability

    # Add 4-row gap between epochs
    rows.extend([[], [], [], []])

    df = pd.DataFrame(rows)
    df.to_csv(save_path, mode='a', index=False, header=not os.path.exists(save_path))



def train_pipeline(csv_path, epochs=100, save_csv_path="model_outputs.csv", model_ckpt_path="final_model_checkpoint.pth"):
    preprocessor = AccidentPreprocessor()
    full_dataset = AccidentDataset(csv_path, preprocessor)
    
    # Split dataset: 80% train, 20% val
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_dim = full_dataset.X.shape[1]
    model = AccidentPredictionPipeline(input_dim, hidden_dim=128, latent_dim=64)
    
    # ✅ Add weight decay here
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    time_points = torch.linspace(0, 1, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Remove existing CSV file if it exists
    if os.path.exists(save_csv_path):
        os.remove(save_csv_path)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Training:")
        for batch_idx, batch_x in enumerate(train_loader):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, time_points)
            loss = composite_loss(outputs, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"  [Train] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        print("Validation:")
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(val_loader):
                batch_x = batch_x.to(device)
                outputs = model(batch_x, time_points)
                loss = composite_loss(outputs, batch_x)
                val_loss += loss.item()
                print(f"  [Val]   Batch {batch_idx+1}/{len(val_loader)} - Loss: {loss.item():.4f}", end='\r')
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary → Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
    
    # ✅ Save model at the end
    print(f"\nSaving model checkpoint to '{model_ckpt_path}'...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': 128,
        'latent_dim': 64,
        'epochs_trained': epochs,
    }, model_ckpt_path)

    return model


# In[13]:


model = train_pipeline("raw_combined_accidents.csv")


# In[ ]:


class ModelEvaluationMetrics:
    def __init__(self, original, reconstructed, n_clusters=5):
        self.original = np.array(original)
        self.reconstructed = np.array(reconstructed)
        self.n_clusters = n_clusters

    def compute_r2_scores(self):
        return [r2_score(self.original[:, i], self.reconstructed[:, i]) for i in range(self.original.shape[1])]

    def compute_cosine_similarity(self):
        sims = [cosine_similarity([self.original[i]], [self.reconstructed[i]])[0, 0] for i in range(len(self.original))]
        return np.mean(sims)

    def compute_cluster_preservation(self):
        pca = PCA(n_components=10)
        X_orig = pca.fit_transform(self.original)
        X_recon = pca.transform(self.reconstructed)

        kmeans_orig = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X_orig)
        kmeans_recon = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X_recon)

        ari = adjusted_rand_score(kmeans_orig.labels_, kmeans_recon.labels_)
        nmi = normalized_mutual_info_score(kmeans_orig.labels_, kmeans_recon.labels_)
        return ari, nmi

    def compute_spearman_rank(self):
        return [spearmanr(self.original[:, i], self.reconstructed[:, i])[0] for i in range(self.original.shape[1])]

    def compute_tsne_umap(self):
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        umap_reducer = umap.UMAP(random_state=0)

        tsne_orig = tsne.fit_transform(self.original)
        tsne_recon = tsne.fit_transform(self.reconstructed)

        umap_orig = umap_reducer.fit_transform(self.original)
        umap_recon = umap_reducer.transform(self.reconstructed)

        return tsne_orig, tsne_recon, umap_orig, umap_recon

    def plot_feature_error_heatmap(self):
        errors = np.abs(self.original - self.reconstructed)
        plt.figure(figsize=(12, 6))
        sns.heatmap(errors, cmap="magma", cbar=True)
        plt.title("Feature-wise Absolute Reconstruction Errors")
        plt.xlabel("Feature Index")
        plt.ylabel("Sample Index")
        plt.tight_layout()
        plt.show()

    def evaluate_all(self):
        metrics = {}
        metrics['r2_per_feature'] = self.compute_r2_scores()
        metrics['cosine_similarity'] = self.compute_cosine_similarity()
        metrics['ari'], metrics['nmi'] = self.compute_cluster_preservation()
        metrics['spearman_per_feature'] = self.compute_spearman_rank()
        return metrics


# In[9]:


def evaluate_model(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    original_all = []
    reconstructed_all = []

    time_points = torch.linspace(0, 1, 2).to(device)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch, time_points)
            original_all.append(batch.cpu().numpy())
            reconstructed_all.append(outputs["reconstructed_input"].cpu().numpy())

    original_all = np.concatenate(original_all, axis=0)
    reconstructed_all = np.concatenate(reconstructed_all, axis=0)

    # Evaluate using metrics
    evaluator = ModelEvaluationMetrics(original_all, reconstructed_all)
    results = evaluator.evaluate_all()

    print("\n=== Model Evaluation Metrics ===")
    print(f"Average R²: {np.mean(results['r2_per_feature']):.4f}")
    print(f"Cosine Similarity: {results['cosine_similarity']:.4f}")
    print(f"Adjusted Rand Index (ARI): {results['ari']:.4f}")
    print(f"Normalized Mutual Info (NMI): {results['nmi']:.4f}")
    print(f"Average Spearman Correlation: {np.nanmean(results['spearman_per_feature']):.4f}")

    # Plot visualizations
    evaluator.plot_feature_error_heatmap()
    tsne_orig, tsne_recon, umap_orig, umap_recon = evaluator.compute_tsne_umap()

    # t-SNE Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("t-SNE: Original vs Reconstructed")
    plt.scatter(tsne_orig[:, 0], tsne_orig[:, 1], label='Original', alpha=0.5)
    plt.scatter(tsne_recon[:, 0], tsne_recon[:, 1], label='Reconstructed', alpha=0.5)
    plt.legend()

    # UMAP Plot
    plt.subplot(1, 2, 2)
    plt.title("UMAP: Original vs Reconstructed")
    plt.scatter(umap_orig[:, 0], umap_orig[:, 1], label='Original', alpha=0.5)
    plt.scatter(umap_recon[:, 0], umap_recon[:, 1], label='Reconstructed', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor = AccidentPreprocessor()
full_dataset = AccidentDataset("raw_combined_accidents.csv", preprocessor)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size])

evaluate_model(model, val_dataset, device)


# In[ ]:




