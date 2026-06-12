"""
HW1: Policy Gradient + Behaviour Cloning on CartPole-v1

Each PG variant is a separate loss callable composed in the training loop.
Run with: python cartpole_pg.py --method {vpg,vpg_avg,vpg_value,rloo,vpg_entropy,vpg_value_entropy,bc,all}
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    max_steps: int = 500

    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"  # NOTE: tanh >> ReLU for PG empirically

    lr_policy: float = 1e-3
    lr_value: float = 3e-3
    max_episodes: int = 1500
    episodes_per_update: int = 8  # also K for RLOO
    gradient_clip: float = 0.5
    normalize_returns: bool = True

    entropy_beta: float = 0.01
    entropy_schedule: str = "linear"  # "constant", "linear", "cosine"
    entropy_min_beta: float = 0.001

    K: int = 8

    eval_every: int = 50
    eval_episodes: int = 20
    solve_threshold: float = 475.0
    solve_window: int = 5

    bc_num_expert_episodes: int = 100
    bc_epochs: int = 30
    bc_lr: float = 1e-3
    bc_batch_size: int = 256

    seed: int = 42
    log_dir: str = "runs"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}



class PolicyNetwork(nn.Module):
    """MLP policy pi_theta(a|s) -> logits for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int],
                 activation: str = "tanh"):
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU

        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> Categorical:
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        dist = self.forward(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Batched log_probs + entropy for state-action pairs."""
        dist = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """MLP value function V_phi(s) -> scalar."""

    def __init__(self, state_dim: int, hidden_dims: List[int],
                 activation: str = "tanh"):
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU

        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)



@dataclass
class Trajectory:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    total_reward: float = 0.0


def collect_trajectory(env: gym.Env, policy: PolicyNetwork) -> Trajectory:
    traj = Trajectory()
    state, _ = env.reset()
    done = False

    while not done:
        action, log_prob = policy.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        traj.states.append(state)
        traj.actions.append(action)
        traj.rewards.append(reward)
        traj.log_probs.append(log_prob)

        state = next_state

    traj.total_reward = sum(traj.rewards)
    return traj


def collect_batch(env: gym.Env, policy: PolicyNetwork, n_episodes: int) -> List[Trajectory]:
    return [collect_trajectory(env, policy) for _ in range(n_episodes)]



def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """G_t = sum y^{t'-t} r_{t'}, reverse accumulation O(T)."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return torch.FloatTensor(returns[::-1])


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)



def loss_vanilla_pg(
    policy: PolicyNetwork,
    batch: List[Trajectory],
    config: Config,
    value_net: Optional[ValueNetwork] = None,
    **kwargs
) -> Tuple[torch.Tensor, dict]:
    """REINFORCE: L = -sum log pi(a|s) * G_t"""
    all_log_probs = []
    all_returns = []

    for traj in batch:
        returns = compute_returns(traj.rewards, config.gamma)
        states = torch.FloatTensor(np.array(traj.states))
        actions = torch.LongTensor(traj.actions)
        log_probs, _ = policy.evaluate_actions(states, actions)
        all_log_probs.append(log_probs)
        all_returns.append(returns)

    all_log_probs = torch.cat(all_log_probs)
    all_returns = torch.cat(all_returns)

    if config.normalize_returns:
        all_returns = normalize(all_returns)

    policy_loss = -(all_log_probs * all_returns.detach()).mean()

    info = {"policy_loss": policy_loss.item()}
    return policy_loss, info


def loss_pg_avg_baseline(
    policy: PolicyNetwork,
    batch: List[Trajectory],
    config: Config,
    value_net: Optional[ValueNetwork] = None,
    ema_reward: float = 0.0,
    **kwargs
) -> Tuple[torch.Tensor, dict]:
    """PG with batch-mean reward baseline: A_t = G_t - mean(R)."""
    all_log_probs = []
    all_advantages = []

    batch_returns = [traj.total_reward for traj in batch]
    baseline = np.mean(batch_returns)

    for traj in batch:
        returns = compute_returns(traj.rewards, config.gamma)
        states = torch.FloatTensor(np.array(traj.states))
        actions = torch.LongTensor(traj.actions)
        log_probs, _ = policy.evaluate_actions(states, actions)
        advantages = returns - baseline

        all_log_probs.append(log_probs)
        all_advantages.append(advantages)

    all_log_probs = torch.cat(all_log_probs)
    all_advantages = torch.cat(all_advantages)

    if config.normalize_returns:
        all_advantages = normalize(all_advantages)

    policy_loss = -(all_log_probs * all_advantages.detach()).mean()

    info = {
        "policy_loss": policy_loss.item(),
        "baseline": baseline,
    }
    return policy_loss, info


def loss_pg_value_baseline(
    policy: PolicyNetwork,
    batch: List[Trajectory],
    config: Config,
    value_net: ValueNetwork = None,
    **kwargs
) -> Tuple[torch.Tensor, dict]:
    """PG with learned value baseline: A_t = G_t - V_phi(s_t). Returns combined loss."""
    assert value_net is not None, "Value network required for value baseline"

    all_log_probs = []
    all_advantages = []
    all_returns = []
    all_values = []

    for traj in batch:
        returns = compute_returns(traj.rewards, config.gamma)
        states = torch.FloatTensor(np.array(traj.states))
        actions = torch.LongTensor(traj.actions)
        log_probs, _ = policy.evaluate_actions(states, actions)

        values = value_net(states)
        advantages = returns - values.detach()

        all_log_probs.append(log_probs)
        all_advantages.append(advantages)
        all_returns.append(returns)
        all_values.append(values)

    all_log_probs = torch.cat(all_log_probs)
    all_advantages = torch.cat(all_advantages)
    all_returns = torch.cat(all_returns)
    all_values = torch.cat(all_values)

    if config.normalize_returns:
        all_advantages = normalize(all_advantages)

    policy_loss = -(all_log_probs * all_advantages.detach()).mean()

    value_loss = nn.functional.mse_loss(all_values, all_returns.detach())

    total_loss = policy_loss + 0.5 * value_loss

    info = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_advantage": all_advantages.mean().item(),
    }
    return total_loss, info


def loss_rloo(
    policy: PolicyNetwork,
    batch: List[Trajectory],
    config: Config,
    value_net: Optional[ValueNetwork] = None,
    **kwargs
) -> Tuple[torch.Tensor, dict]:
    """RLOO: b_k = (1/(K-1)) sum_{j!=k} R_j."""
    K = len(batch)
    assert K >= 2, "RLOO requires at least 2 trajectories"

    episode_returns = torch.FloatTensor([traj.total_reward for traj in batch])
    total_sum = episode_returns.sum()

    all_log_probs = []
    all_advantages = []

    for k, traj in enumerate(batch):
        returns = compute_returns(traj.rewards, config.gamma)
        states = torch.FloatTensor(np.array(traj.states))
        actions = torch.LongTensor(traj.actions)
        log_probs, _ = policy.evaluate_actions(states, actions)

        b_k = (total_sum - episode_returns[k]) / (K - 1)
        advantages = returns - b_k.item()

        all_log_probs.append(log_probs)
        all_advantages.append(advantages)

    all_log_probs = torch.cat(all_log_probs)
    all_advantages = torch.cat(all_advantages)

    if config.normalize_returns:
        all_advantages = normalize(all_advantages)

    policy_loss = -(all_log_probs * all_advantages.detach()).mean()

    info = {
        "policy_loss": policy_loss.item(),
        "mean_loo_baseline": ((total_sum - episode_returns) / (K - 1)).mean().item(),
    }
    return policy_loss, info


def add_entropy_regularization(
    base_loss_fn: Callable,
    policy: PolicyNetwork,
    batch: List[Trajectory],
    config: Config,
    current_episode: int,
    **kwargs
) -> Tuple[torch.Tensor, dict]:
    """Wraps any base loss and adds -beta*H(pi). Composable with all PG losses."""
    base_loss, info = base_loss_fn(policy=policy, batch=batch, config=config, **kwargs)

    all_states = []
    for traj in batch:
        all_states.append(torch.FloatTensor(np.array(traj.states)))
    all_states = torch.cat(all_states)

    dist = policy(all_states)
    entropy = dist.entropy().mean()

    beta = _schedule_beta(config, current_episode)
    total_loss = base_loss - beta * entropy

    info["entropy"] = entropy.item()
    info["entropy_beta"] = beta
    info["entropy_bonus"] = (beta * entropy).item()
    return total_loss, info


def _schedule_beta(config: Config, episode: int) -> float:
    """Compute entropy coefficient based on schedule."""
    progress = min(episode / config.max_episodes, 1.0)

    if config.entropy_schedule == "constant":
        return config.entropy_beta
    elif config.entropy_schedule == "linear":
        return config.entropy_beta + (config.entropy_min_beta - config.entropy_beta) * progress
    elif config.entropy_schedule == "cosine":
        return config.entropy_min_beta + 0.5 * (config.entropy_beta - config.entropy_min_beta) * \
               (1 + np.cos(np.pi * progress))
    else:
        return config.entropy_beta



def train(
    loss_fn: Callable,
    config: Config,
    use_value_net: bool = False,
    use_entropy: bool = False,
    method_name: str = "vpg",
) -> Tuple[PolicyNetwork, dict]:
    """Generic training loop — accepts any PG loss callable."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim, config.hidden_dims, config.activation)
    policy_optimizer = optim.Adam(policy.parameters(), lr=config.lr_policy)

    # NOTE: tried shared trunk for policy+value, gradient interference made it worse
    value_net = None
    value_optimizer = None
    if use_value_net:
        value_net = ValueNetwork(state_dim, config.hidden_dims, config.activation)
        value_optimizer = optim.Adam(value_net.parameters(), lr=config.lr_value)

    history = {
        "episode_rewards": [],
        "eval_rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "gradient_norms": [],
        "grad_variances": [],
        "grad_l1_norms": [],
        "time_per_update": [],
    }

    episode_count = 0
    update_count = 0
    solved = False
    eval_window = deque(maxlen=config.solve_window)
    train_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"Training: {method_name}")
    print(f"{'='*60}\n")

    while episode_count < config.max_episodes:
        batch = collect_batch(env, policy, config.episodes_per_update)
        episode_count += len(batch)

        batch_rewards = [traj.total_reward for traj in batch]
        history["episode_rewards"].extend(batch_rewards)

        if use_entropy:
            loss, info = add_entropy_regularization(
                base_loss_fn=loss_fn,
                policy=policy,
                batch=batch,
                config=config,
                current_episode=episode_count,
                value_net=value_net,
            )
        else:
            loss, info = loss_fn(
                policy=policy,
                batch=batch,
                config=config,
                value_net=value_net,
            )

        update_start = time.perf_counter()
        policy_optimizer.zero_grad()
        if value_optimizer:
            value_optimizer.zero_grad()
        loss.backward()

        grad_vec = torch.cat([
            p.grad.flatten()
            for p in policy.parameters()
            if p.grad is not None
        ])
        history["grad_variances"].append(torch.var(grad_vec).item())
        history["grad_l1_norms"].append(torch.mean(torch.abs(grad_vec)).item())

        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
        history["gradient_norms"].append(grad_norm.item())

        policy_optimizer.step()
        if value_optimizer:
            value_optimizer.step()

        history["time_per_update"].append(time.perf_counter() - update_start)
        history["policy_losses"].append(info.get("policy_loss", 0))
        history["value_losses"].append(info.get("value_loss", 0))
        history["entropies"].append(info.get("entropy", 0))
        update_count += 1

        if episode_count % config.eval_every < config.episodes_per_update:
            mean_reward, std_reward = evaluate(policy, config)
            history["eval_rewards"].append({
                "episode": episode_count,
                "mean": mean_reward,
                "std": std_reward,
            })
            eval_window.append(mean_reward)

            print(f"Episode {episode_count:5d} | "
                  f"Train reward: {np.mean(batch_rewards):7.1f} | "
                  f"Eval: {mean_reward:7.1f} ± {std_reward:5.1f} | "
                  f"Loss: {info.get('policy_loss', 0):8.4f}"
                  + (f" | Entropy: {info.get('entropy', 0):.4f}" if use_entropy else "")
                  + (f" | V_loss: {info.get('value_loss', 0):.4f}" if use_value_net else ""))

            if len(eval_window) == config.solve_window and \
               all(r >= config.solve_threshold for r in eval_window):
                print(f"\nSolved at episode {episode_count}!")
                solved = True
                break

    env.close()

    history["solved"] = solved
    history["total_episodes"] = episode_count
    history["method"] = method_name
    history["wall_time_total"] = time.perf_counter() - train_start

    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    save_dir = os.path.join(ROOT, "results", "raw")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{safe_name}_seed{config.seed}.json")

    def _to_serializable(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    record = {
        "config": config.to_dict(),
        "history": {k: [_to_serializable(x) for x in v] if isinstance(v, list) else _to_serializable(v)
                    for k, v in history.items()},
    }
    with open(save_path, "w") as f:
        json.dump(record, f, default=str)

    return policy, history



def load_results(results_dir: str = None) -> dict:
    """Load all JSON results, grouped by method: {name: [record_per_seed, ...]}."""
    if results_dir is None:
        results_dir = os.path.join(ROOT, "results", "raw")
    grouped = {}
    if not os.path.isdir(results_dir):
        return grouped
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            record = json.load(f)
        if "history" not in record:
            continue
        method = record["history"].get("method", fname)
        grouped.setdefault(method, []).append(record)
    return grouped



def evaluate(
    policy: PolicyNetwork,
    config: Config,
    n_episodes: Optional[int] = None,
    render: bool = False,
) -> Tuple[float, float]:
    """Greedy (argmax) eval over N episodes -> (mean, std)."""
    n_episodes = n_episodes or config.eval_episodes
    env = gym.make(config.env_name, render_mode="human" if render else None)
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                dist = policy(state_t)
                action = dist.probs.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards), np.std(rewards)



def collect_expert_data(
    expert_policy: PolicyNetwork,
    config: Config,
    n_episodes: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (state, action) pairs from deterministic expert -> (N, 4), (N,)."""
    env = gym.make(config.env_name)
    all_states = []
    all_actions = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=config.seed + ep)
        done = False

        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                dist = expert_policy(state_t)
                action = dist.probs.argmax().item()

            all_states.append(state)
            all_actions.append(action)

            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    states = np.array(all_states)
    actions = np.array(all_actions)
    print(f"Collected {len(states)} transitions from {n_episodes} episodes")
    return states, actions


def train_bc(
    states: np.ndarray,
    actions: np.ndarray,
    config: Config,
    subset_size: Optional[int] = None,
) -> Tuple[PolicyNetwork, dict]:
    """Supervised classification on expert (state, action) pairs."""
    if subset_size is not None:
        indices = np.random.choice(len(states), size=min(subset_size, len(states)), replace=False)
        states = states[indices]
        actions = actions[indices]

    n = len(states)
    n_train = int(0.9 * n)
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions)

    state_dim = states.shape[1]
    action_dim = int(actions.max()) + 1
    bc_policy = PolicyNetwork(state_dim, action_dim, config.hidden_dims, config.activation)
    optimizer = optim.Adam(bc_policy.parameters(), lr=config.bc_lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(config.bc_epochs):
        bc_policy.train()
        perm_train = np.random.permutation(n_train)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, config.bc_batch_size):
            idx = train_idx[perm_train[i:i + config.bc_batch_size]]
            batch_states = states_t[idx]
            batch_actions = actions_t[idx]

            logits = bc_policy.net(batch_states)  # bypass Categorical for CE loss
            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        bc_policy.eval()
        with torch.no_grad():
            val_logits = bc_policy.net(states_t[val_idx])
            val_loss = criterion(val_logits, actions_t[val_idx]).item()
            val_acc = (val_logits.argmax(dim=1) == actions_t[val_idx]).float().mean().item()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train loss: {epoch_loss/n_batches:.4f} | "
                  f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.3f}")

    return bc_policy, history



def bc_failure_experiments(
    expert_policy: PolicyNetwork,
    config: Config,
):
    """Dataset size, narrow distribution, noisy expert, and state distribution shift."""
    print("\n" + "="*60)
    print("BEHAVIOUR CLONING FAILURE EXPERIMENTS")
    print("="*60)

    all_states, all_actions = collect_expert_data(expert_policy, config, n_episodes=100)

    results = {}

    print("\n--- Exp 1: Dataset Size Ablation ---")
    for n_transitions in [500, 1000, 5000, 10000, len(all_states)]:
        bc_policy, bc_hist = train_bc(all_states, all_actions, config, subset_size=n_transitions)
        mean_r, std_r = evaluate(bc_policy, config, n_episodes=50)
        results[f"size_{n_transitions}"] = {"mean": mean_r, "std": std_r}
        print(f"  N={n_transitions:6d} transitions -> Reward: {mean_r:.1f} ± {std_r:.1f}")

    print("\n--- Exp 2: Narrow State Distribution ---")
    env = gym.make(config.env_name)
    narrow_states, narrow_actions = [], []

    for ep in range(50):
        state, _ = env.reset(seed=42)  # same seed -> narrow initial state coverage
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                dist = expert_policy(state_t)
                action = dist.probs.argmax().item()
            narrow_states.append(state)
            narrow_actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

    narrow_states = np.array(narrow_states)
    narrow_actions = np.array(narrow_actions)
    print(f"  Narrow dataset: {len(narrow_states)} transitions")
    print(f"  State range — cart_pos: [{narrow_states[:,0].min():.3f}, {narrow_states[:,0].max():.3f}]"
          f"  pole_angle: [{narrow_states[:,2].min():.3f}, {narrow_states[:,2].max():.3f}]")

    bc_narrow, _ = train_bc(narrow_states, narrow_actions, config)
    mean_r, std_r = evaluate(bc_narrow, config, n_episodes=50)
    results["narrow_dist"] = {"mean": mean_r, "std": std_r}
    print(f"  Narrow-trained BC -> Reward: {mean_r:.1f} ± {std_r:.1f}")

    print("\n--- Exp 3: Noisy Expert Actions ---")
    for noise_ratio in [0.0, 0.05, 0.1, 0.2, 0.5]:
        noisy_actions = all_actions.copy()
        n_flip = int(noise_ratio * len(noisy_actions))
        flip_idx = np.random.choice(len(noisy_actions), size=n_flip, replace=False)
        noisy_actions[flip_idx] = 1 - noisy_actions[flip_idx]  # Flip 0<->1

        bc_noisy, _ = train_bc(all_states, noisy_actions, config)
        mean_r, std_r = evaluate(bc_noisy, config, n_episodes=50)
        results[f"noise_{noise_ratio}"] = {"mean": mean_r, "std": std_r}
        print(f"  Noise {noise_ratio*100:4.0f}% -> Reward: {mean_r:.1f} ± {std_r:.1f}")

    print("\n--- Exp 4: State Distribution Shift ---")
    bc_full, _ = train_bc(all_states, all_actions, config)
    env = gym.make(config.env_name)

    expert_visited = []
    bc_visited = []

    for _ in range(50):
        state, _ = env.reset()
        done = False
        while not done:
            expert_visited.append(state.copy())
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = expert_policy(state_t).probs.argmax().item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    for _ in range(50):
        state, _ = env.reset()
        done = False
        while not done:
            bc_visited.append(state.copy())
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = bc_full(state_t).probs.argmax().item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    expert_visited = np.array(expert_visited)
    bc_visited = np.array(bc_visited)

    results["state_dist"] = {
        "expert_cart_pos_std": expert_visited[:, 0].std(),
        "bc_cart_pos_std": bc_visited[:, 0].std(),
        "expert_pole_angle_std": expert_visited[:, 2].std(),
        "bc_pole_angle_std": bc_visited[:, 2].std(),
    }
    print(f"  Expert state std (cart_pos, pole_angle): "
          f"({expert_visited[:,0].std():.4f}, {expert_visited[:,2].std():.4f})")
    print(f"  BC     state std (cart_pos, pole_angle): "
          f"({bc_visited[:,0].std():.4f}, {bc_visited[:,2].std():.4f})")

    return results



def plot_training_curves(histories: dict, save_path: str = "learning_curves.png"):
    """Learning curves with +/-1 std shading across seeds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, seed_histories), color in zip(histories.items(), colors):
        min_len = min(len(h["episode_rewards"]) for h in seed_histories)
        rewards = np.array([h["episode_rewards"][:min_len] for h in seed_histories])

        window = 50
        smoothed = np.array([
            np.convolve(r, np.ones(window)/window, mode='valid')
            for r in rewards
        ])

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        episodes = np.arange(len(mean))

        ax.plot(episodes, mean, label=name, color=color)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Policy Gradient Methods — CartPole-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=475, color='gray', linestyle='--', alpha=0.5, label='Solve threshold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")



def run_all_experiments(config: Config, n_seeds: int = 3):
    """Run all PG variants x seeds, then BC failure experiments."""

    methods = {
        "VPG": {
            "loss_fn": loss_vanilla_pg,
            "use_value_net": False,
            "use_entropy": False,
        },
        "VPG + Avg Baseline": {
            "loss_fn": loss_pg_avg_baseline,
            "use_value_net": False,
            "use_entropy": False,
        },
        "VPG + Value Baseline": {
            "loss_fn": loss_pg_value_baseline,
            "use_value_net": True,
            "use_entropy": False,
        },
        "RLOO (K=8)": {
            "loss_fn": loss_rloo,
            "use_value_net": False,
            "use_entropy": False,
        },
        "VPG + Entropy": {
            "loss_fn": loss_vanilla_pg,
            "use_value_net": False,
            "use_entropy": True,
        },
        "VPG + Value + Entropy": {
            "loss_fn": loss_pg_value_baseline,
            "use_value_net": True,
            "use_entropy": True,
        },
    }

    all_histories = {}
    best_policy = None
    best_reward = -float("inf")

    for method_name, method_config in methods.items():
        seed_histories = []

        for seed in range(n_seeds):
            cfg = Config()
            cfg.seed = config.seed + seed

            policy, history = train(
                loss_fn=method_config["loss_fn"],
                config=cfg,
                use_value_net=method_config["use_value_net"],
                use_entropy=method_config["use_entropy"],
                method_name=f"{method_name} (seed={cfg.seed})",
            )
            seed_histories.append(history)

            final_reward = np.mean(history["episode_rewards"][-100:])
            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = policy

        all_histories[method_name] = seed_histories

    plot_training_curves(all_histories)

    print("\n" + "="*80)
    print(f"{'Method':<30} | {'Mean Reward':>12} | {'Std':>8} | {'Solved':>6} | {'Episodes':>8}")
    print("-"*80)
    for name, histories in all_histories.items():
        mean_rewards = [np.mean(h["episode_rewards"][-100:]) for h in histories]
        solved = [h["solved"] for h in histories]
        total_eps = [h["total_episodes"] for h in histories]
        print(f"{name:<30} | {np.mean(mean_rewards):>12.1f} | {np.std(mean_rewards):>8.1f} | "
              f"{sum(solved):>3}/{len(solved):<2} | {np.mean(total_eps):>8.0f}")
    print("="*80)

    if best_policy is not None:
        print(f"\nBest expert reward: {best_reward:.1f}")
        bc_results = bc_failure_experiments(best_policy, config)

    return all_histories, bc_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="all",
                        choices=["vpg", "vpg_avg", "vpg_value", "rloo",
                                 "vpg_entropy", "vpg_value_entropy", "bc", "all"])
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy_beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        K=args.K,
        max_episodes=args.episodes,
        lr_policy=args.lr,
        entropy_beta=args.entropy_beta,
        seed=args.seed,
    )

    if args.method == "all":
        run_all_experiments(config, n_seeds=args.seeds)
    else:
        method_map = {
            "vpg": (loss_vanilla_pg, False, False),
            "vpg_avg": (loss_pg_avg_baseline, False, False),
            "vpg_value": (loss_pg_value_baseline, True, False),
            "rloo": (loss_rloo, False, False),
            "vpg_entropy": (loss_vanilla_pg, False, True),
            "vpg_value_entropy": (loss_pg_value_baseline, True, True),
        }

        if args.method == "bc":
            print("Training expert policy (VPG + Value + Entropy)...")
            expert, _ = train(
                loss_fn=loss_pg_value_baseline,
                config=config,
                use_value_net=True,
                use_entropy=True,
                method_name="Expert (VPG+V+Ent)",
            )
            bc_failure_experiments(expert, config)
        else:
            loss_fn, use_v, use_ent = method_map[args.method]
            policy, history = train(
                loss_fn=loss_fn,
                config=config,
                use_value_net=use_v,
                use_entropy=use_ent,
                method_name=args.method,
            )
            mean_r, std_r = evaluate(policy, config, n_episodes=100)
            print(f"\nFinal evaluation: {mean_r:.1f} ± {std_r:.1f}")
