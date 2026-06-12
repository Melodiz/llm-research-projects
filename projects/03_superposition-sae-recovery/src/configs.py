from dataclasses import dataclass


@dataclass
class ToyModelConfig:
    F: int = 50              # number of ground-truth features
    d: int = 5               # hidden dimension
    alpha: float = 1.0       # power-law exponent for feature probabilities
    # I = 1 (uniform importance, hardcoded for Parts 1-3)
    lr: float = 1e-3
    train_steps: int = 10_000
    batch_size: int = 1024
    seed: int = 42


@dataclass
class SAEConfig:
    F_sae: int = 100         # dictionary size (default: 2*F)
    l0_coeff: float = 0.1    # L1 sparsity coefficient (called l0 in assignment)
    lr: float = 1e-3
    train_steps: int = 20_000
    batch_size: int = 1024
    l1_warmup_frac: float = 0.1  # fraction of steps for L1 warmup
    seed: int = 42
