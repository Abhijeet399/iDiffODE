import torch
import torch.nn as nn
from torchdiffeq import odeint


# ----------------------------
# ODE functions / encoders
# ----------------------------

class TemporalODEFuncMLP(nn.Module):
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
        # x: [batch, features]
        t_tensor = torch.ones_like(x[:, :1]) * t
        x_with_time = torch.cat([x, t_tensor], dim=1)
        return self.net(x_with_time)


class TemporalODERNNFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim + 1, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, x):
        # x: [batch, features]
        t_tensor = torch.ones_like(x[:, :1]) * t
        x_with_time = torch.cat([x, t_tensor], dim=1)  # [batch, input_dim+1]
        x_seq = x_with_time.unsqueeze(1)  # [batch, seq=1, input_dim+1]
        rnn_out, _ = self.rnn(x_seq)  # [batch, seq=1, hidden_dim]
        out = self.output_layer(rnn_out.squeeze(1))  # [batch, input_dim]
        return out


class TemporalODETransformerFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead=8, num_layers=2):
        super().__init__()
        self.token_proj = nn.Linear(input_dim + 1, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, t, x):
        B, D = x.size()
        t_tensor = torch.ones(B, 1, device=x.device) * t
        x_with_time = torch.cat([x, t_tensor], dim=1)  # [B, D+1]
        x_proj = self.token_proj(x_with_time).unsqueeze(1)  # [B, 1, D]
        attended = self.transformer(x_proj)  # [B, 1, D]
        return attended.squeeze(1)


# ----------------------------
# Encoder wrappers
# ----------------------------

class NeuralODEEncoderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ode_func = TemporalODEFuncMLP(hidden_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, time_points):
        h = self.input_proj(x)
        h = odeint(self.ode_func, h, time_points, method='dopri5')[-1]
        latent = self.latent_proj(h)
        return latent


class NeuralODEEncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ode_func = TemporalODERNNFunc(hidden_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, time_points):
        h = self.input_proj(x)
        h = odeint(self.ode_func, h, time_points, method='dopri5')[-1]
        latent = self.latent_proj(h)
        return latent


class NeuralODEEncoderTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, nhead=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.ode_func = TemporalODETransformerFunc(input_dim, hidden_dim, nhead=nhead, num_layers=num_layers)
        self.latent_proj = nn.Linear(input_dim, latent_dim)

    def forward(self, x, time_points):
        h = self.input_proj(x)
        h = odeint(self.ode_func, h, time_points, method='dopri5')[-1]
        latent = self.latent_proj(h)
        return latent


# ----------------------------
# Diffusion + inverse network
# ----------------------------

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
        # t is a tensor of indices
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1).to(x.device)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1).to(x.device)
        epsilon = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * epsilon, epsilon

    def forward(self, x):
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device)
        noisy_x, noise = self.add_noise(x, t)
        t_norm = t.float() / self.timesteps
        predicted_noise = self.noise_predictor(torch.cat([noisy_x, t_norm.unsqueeze(1)], dim=1))
        return noisy_x, noise, predicted_noise

    def sample(self, x, steps=100):
        # iterative denoising as in the original snippet
        for t in reversed(range(steps)):
            t_tensor = torch.full((x.size(0), 1), t/steps, device=x.device)
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


class InverseResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
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
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return x


# ----------------------------
# Full pipeline wrapper
# ----------------------------

class AccidentPredictionPipeline(nn.Module):
    """
    model_type: 'mlp', 'rnn', 'transformer'
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, model_type='mlp', transformer_heads=8, transformer_layers=2, diffusion_timesteps=1000):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == 'mlp':
            self.encoder = NeuralODEEncoderMLP(input_dim, hidden_dim, latent_dim)
        elif self.model_type == 'rnn':
            self.encoder = NeuralODEEncoderRNN(input_dim, hidden_dim, latent_dim)
        elif self.model_type == 'transformer':
            self.encoder = NeuralODEEncoderTransformer(input_dim, hidden_dim, latent_dim, nhead=transformer_heads, num_layers=transformer_layers)
        else:
            raise ValueError(f"Unknown model_type {model_type}")

        self.diffusion = AccidentDiffusion(latent_dim, hidden_dim, timesteps=diffusion_timesteps)
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
