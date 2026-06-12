import torch
from toy_model import generate_batch, compute_feature_probabilities
from configs import ToyModelConfig


def explained_variance(x, x_hat):
    """EV = 1 - sqrt(Var(residual) / Var(x)). Stricter than R^2."""
    residual_var = (x - x_hat).var()
    input_var = x.var()
    return (1.0 - (residual_var / input_var).sqrt()).item()


def mean_max_cosine_similarity(W_true, W_learned):
    """MMCS: for each GT feature, find best-matching learned direction.
    Returns (mmcs_scalar, assignment_indices).
    """
    W_true_n = W_true / W_true.norm(dim=1, keepdim=True)
    W_learned_n = W_learned / W_learned.norm(dim=1, keepdim=True)
    cos_sim = (W_true_n @ W_learned_n.T).abs()
    max_cos, assignments = cos_sim.max(dim=1)
    return max_cos.mean().item(), assignments


def weighted_mmcs(W_true, W_learned):
    """MMCS weighted by ||W_i|| / max(||W||).
    Prevents zero-norm features from inflating the score.
    Returns (wmmcs_scalar, assignment_indices).
    """
    W_true_n = W_true / W_true.norm(dim=1, keepdim=True)
    W_learned_n = W_learned / W_learned.norm(dim=1, keepdim=True)
    cos_sim = (W_true_n @ W_learned_n.T).abs()
    max_cos, assignments = cos_sim.max(dim=1)

    norms = W_true.norm(dim=1)
    weights = norms / norms.max()
    wmmcs = (weights * max_cos).sum() / weights.sum()
    return wmmcs.item(), assignments


def fraction_dead_latents(activations, threshold=0.0):
    """Fraction of SAE latents that never fire."""
    ever_active = (activations > threshold).any(dim=0)
    return (1.0 - ever_active.float().mean()).item()


def cosine_similarity_matrix(W):
    """Pairwise cosine sim of rows of W. Returns (F, F) matrix."""
    W_n = W / W.norm(dim=1, keepdim=True)
    return W_n @ W_n.T


def feature_recovery_rate(W_true, W_learned, threshold=0.9):
    """Fraction of GT features with max cosine > threshold."""
    W_true_n = W_true / W_true.norm(dim=1, keepdim=True)
    W_learned_n = W_learned / W_learned.norm(dim=1, keepdim=True)
    cos_sim = (W_true_n @ W_learned_n.T).abs()
    max_cos, _ = cos_sim.max(dim=1)
    return (max_cos >= threshold).float().mean().item()


def l0_sparsity(activations):
    """Mean number of nonzero entries per sample."""
    return (activations > 0).float().sum(dim=1).mean().item()


def feature_dimensionality(W):
    """D_i = ||W_i||^2 / sum_j (W_hat_j . W_i)^2. Returns (F,) tensor."""
    norms = W.norm(dim=1, keepdim=True)
    W_hat = W / norms
    dots = (W_hat @ W.T) ** 2
    denom = dots.sum(dim=0)
    numer = (norms.squeeze(1)) ** 2
    return numer / denom


def frequency_recovery(p_true, W_gt, W_dec, W_enc, b_enc, W_toy, alpha, F,
                       batch_size=10000):
    """Compare true feature probs with SAE latent activation rates.
    Returns (true_freqs, sae_freqs), both shape (F,).
    """
    device = W_gt.device

    _, assignments = mean_max_cosine_similarity(W_gt, W_dec.T)

    cfg = ToyModelConfig(F=F, alpha=alpha)
    torch.manual_seed(0)
    x = generate_batch(cfg, batch_size, device)
    with torch.no_grad():
        h = (W_toy.T @ x.T).T
        z = torch.relu(h @ W_enc.T + b_enc)

    sae_freqs = (z[:, assignments] > 0).float().mean(dim=0)
    true_freqs = p_true.to(device)

    return true_freqs, sae_freqs
