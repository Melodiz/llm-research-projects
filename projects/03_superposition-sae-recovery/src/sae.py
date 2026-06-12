import torch
import torch.nn as nn
from configs import SAEConfig, ToyModelConfig
from toy_model import ToyModel


class SAE(nn.Module):
    """Sparse autoencoder: h -> ReLU(W_enc h + b_enc) -> W_dec z + b_dec."""

    def __init__(self, d, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.randn(cfg.F_sae, d, device=device) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(cfg.F_sae, device=device))
        self.W_dec = nn.Parameter(torch.randn(d, cfg.F_sae, device=device) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d, device=device))
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True))

    def forward(self, h):
        """Returns (h_hat, z)."""
        z = self.encode(h)                    # (batch, F_sae)
        h_hat = z @ self.W_dec.T + self.b_dec  # (batch, d)
        return h_hat, z

    def encode(self, h):
        return torch.relu(h @ self.W_enc.T + self.b_enc)


def sae_loss(h, h_hat, activations, l0_coeff, step, total_steps, warmup_frac):
    """MSE + L1 with linear warmup. Returns (loss, {"mse": ..., "l1": ...})."""
    mse = ((h - h_hat) ** 2).mean()
    l1 = activations.abs().mean()
    effective_l0 = l0_coeff * min(1.0, step / (warmup_frac * total_steps))
    loss = mse + effective_l0 * l1
    return loss, {"mse": mse.item(), "l1": l1.item()}


def collect_hidden_activations(model, toy_cfg, n_samples, device, seed=0):
    """Generate data and return hidden activations h = W^T f."""
    from toy_model import generate_batch
    torch.manual_seed(seed)
    with torch.no_grad():
        x = generate_batch(toy_cfg, n_samples, device)
        h = model.encode(x)
    return h


def train_sae(model, toy_cfg, sae_cfg, device):
    """Train SAE on frozen toy model hidden states. Returns (sae, losses)."""
    from toy_model import generate_batch
    torch.manual_seed(sae_cfg.seed)

    sae = SAE(toy_cfg.d, sae_cfg, device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=sae_cfg.lr)
    losses = []

    model.eval()
    for step in range(sae_cfg.train_steps):
        with torch.no_grad():
            x = generate_batch(toy_cfg, sae_cfg.batch_size, device)
            h = model.encode(x)

        h_hat, z = sae(h)
        loss, components = sae_loss(
            h, h_hat, z,
            sae_cfg.l0_coeff, step, sae_cfg.train_steps, sae_cfg.l1_warmup_frac,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # normalize W_dec columns to unit norm
        with torch.no_grad():
            sae.W_dec.div_(sae.W_dec.norm(dim=0, keepdim=True))

        losses.append(loss.item())
        if step % 100 == 0:
            print(f"Step {step:>5d} | Loss: {loss.item():.6f} "
                  f"| MSE: {components['mse']:.6f} | L1: {components['l1']:.6f}")

    return sae, losses
