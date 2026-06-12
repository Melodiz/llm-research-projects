import torch
import torch.nn as nn
from configs import ToyModelConfig


class ToyModel(nn.Module):
    """Single-layer linear autoencoder: f -> ReLU(W @ W^T @ f + b).
    Learns superposition when F > d.
    """

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.W = nn.Parameter(torch.randn(cfg.F, cfg.d, device=device) * 0.01)
        self.b = nn.Parameter(torch.zeros(cfg.F, device=device))

    def forward(self, x):
        # h = W^T @ f^T -> (d, batch), then W @ h -> (F, batch)
        h = self.W.T @ x.T          # (d, batch)
        recon = self.W @ h           # (F, batch)
        recon = recon.T + self.b     # (batch, F)
        return torch.relu(recon)

    def encode(self, x):
        return (self.W.T @ x.T).T  # (batch, d)


def generate_batch(cfg, batch_size, device, seed=None):
    """Sample sparse feature vectors with power-law activation probabilities."""
    if seed is not None:
        torch.manual_seed(seed)
    probs = compute_feature_probabilities(cfg.F, cfg.alpha).to(device)
    mask = torch.bernoulli(probs.unsqueeze(0).expand(batch_size, -1))  # (batch, F)
    magnitudes = torch.rand(batch_size, cfg.F, device=device)          # U(0,1)
    return mask * magnitudes


def compute_feature_probabilities(F, alpha):
    """p_i = C * i^{-alpha}, normalized so sum(p) = 1, clipped to [0,1]."""
    indices = torch.arange(1, F + 1, dtype=torch.float32)
    probs = indices.pow(-alpha)
    C = 1.0 / probs.sum()
    probs = (C * probs).clamp(0.0, 1.0)
    return probs


def importance_weighted_mse(x, x_hat):
    """MSE loss (uniform importance I=1)."""
    return ((x - x_hat) ** 2).mean()


def train_toy_model(cfg, device):
    """Train from scratch. Returns (model, losses)."""
    torch.manual_seed(cfg.seed)
    model = ToyModel(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses = []

    for step in range(cfg.train_steps):
        x = generate_batch(cfg, cfg.batch_size, device)
        x_hat = model(x)
        loss = importance_weighted_mse(x, x_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 100 == 0:
            print(f"Step {step:>5d} | Loss: {loss.item():.6f}")

    return model, losses
