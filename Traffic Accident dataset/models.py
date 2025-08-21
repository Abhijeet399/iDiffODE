import torch
import torch.nn as nn
from torchdiffeq import odeint

# ----- Encoders / ODE functions -----

class TemporalODEFuncMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_in)
        )
    def forward(self, t, x):
        t_tensor = torch.ones_like(x[:, :1]) * t
        x_time = torch.cat([x, t_tensor], dim=1)
        return self.net(x_time)


class TemporalODEFuncRNN(nn.Module):
    def __init__(self, dim_in, hidden_dim, rnn_type='GRU'):
        super().__init__()
        # We'll project (x + t) as sequence length 1 -> run RNN -> map back
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(dim_in + 1, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(dim_in + 1, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, dim_in)
    def forward(self, t, x):
        t_tensor = torch.ones_like(x[:, :1]) * t
        x_time = torch.cat([x, t_tensor], dim=1).unsqueeze(1)  # [B, 1, D+1]
        out, _ = self.rnn(x_time)
        return self.out(out.squeeze(1))


class TemporalODEFuncTransformer(nn.Module):
    def __init__(self, dim_in, hidden_dim, nhead=4, num_layers=2):
        super().__init__()
        self.token_proj = nn.Linear(dim_in + 1, dim_in)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, t, x):
        t_tensor = torch.ones_like(x[:, :1]) * t
        x_time = torch.cat([x, t_tensor], dim=1)
        x_proj = self.token_proj(x_time).unsqueeze(1)  # [B, 1, D]
        out = self.transformer(x_proj)
        return out.squeeze(1)


# ----- Neural ODE Encoder variants -----

class NeuralODEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, ode_func):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim)  # identity-like proj
        self.ode_func = ode_func
        self.latent_proj = nn.Linear(input_dim, latent_dim)

    def forward(self, x, time_points):
        h = self.input_proj(x)
        h = odeint(self.ode_func, h, time_points, method='dopri5')[-1]
        latent = self.latent_proj(h)
        return latent


# ----- Diffusion + InverseResNet as before (kept modular) -----

class AccidentDiffusion(nn.Module):
    def __init__(self, latent_dim, hidden_dim, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
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
        eps = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * eps, eps
    def forward(self, x):
        b = x.size(0)
        t = torch.randint(0, self.timesteps, (b,)).to(x.device)
        noisy, noise = self.add_noise(x, t)
        t_norm = t.float() / self.timesteps
        pred_noise = self.noise_predictor(torch.cat([noisy, t_norm.unsqueeze(1)], dim=1))
        return noisy, noise, pred_noise
    def sample(self, x, steps=100):
        for t in reversed(range(steps)):
            t_tensor = torch.full((x.size(0), 1), t/steps).to(x.device)
            noise_pred = self.noise_predictor(torch.cat([x, t_tensor], dim=1))
            alpha_t = self.alpha[t]; alpha_bar_t = self.alpha_bar[t]; beta_t = self.beta[t]
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) / torch.sqrt(alpha_t)
            x += torch.sqrt(beta_t) * noise
        return x


class InverseResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, y, n_iter=10):
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
        for b in self.blocks:
            x = b(x)
        return self.final(x)


# ----- Pipeline factory -----

class AccidentPredictionPipeline(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, model_type='mlp', transformer_cfg=None):
        super().__init__()
        model_type = model_type.lower()
        if model_type == 'mlp':
            ode_func = TemporalODEFuncMLP(hidden_dim, hidden_dim)
            encoder = NeuralODEEncoder(input_dim, hidden_dim, latent_dim, ode_func)
        elif model_type == 'rnn':
            ode_func = TemporalODEFuncRNN(hidden_dim, hidden_dim)
            encoder = NeuralODEEncoder(input_dim, hidden_dim, latent_dim, ode_func)
        elif model_type == 'transformer':
            cfg = transformer_cfg or {'nhead':4, 'num_layers':2}
            ode_func = TemporalODEFuncTransformer(hidden_dim, hidden_dim, nhead=cfg['nhead'], num_layers=cfg['num_layers'])
            encoder = NeuralODEEncoder(input_dim, hidden_dim, latent_dim, ode_func)
        else:
            raise ValueError("Unknown model_type")

        self.encoder = encoder
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
