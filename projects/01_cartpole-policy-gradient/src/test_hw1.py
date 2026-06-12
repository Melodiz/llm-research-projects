"""test_hw1.py — End-to-end verification for HW1 submission."""

import json
import os
import sys
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(ROOT, "results", "raw")
FIGURES_DIR = os.path.join(ROOT, "figures")
REPORT_PATH = os.path.join(ROOT, "report", "report.pdf")
SUMMARY_PATH = os.path.join(ROOT, "results", "summary.json")

passed = []
failed = []


def run_test(name, fn):
    try:
        fn()
        passed.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        failed.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        print()



def test_imports():
    from cartpole_pg import (
        PolicyNetwork, ValueNetwork, Config,
        collect_trajectory, compute_returns,
        loss_vanilla_pg, loss_pg_avg_baseline,
        loss_pg_value_baseline, loss_rloo,
        add_entropy_regularization,
        train, evaluate, train_bc,
    )
    assert callable(collect_trajectory)
    assert callable(compute_returns)
    assert callable(loss_vanilla_pg)
    assert callable(loss_pg_avg_baseline)
    assert callable(loss_pg_value_baseline)
    assert callable(loss_rloo)
    assert callable(add_entropy_regularization)
    assert callable(train)
    assert callable(evaluate)
    assert callable(train_bc)
    assert isinstance(PolicyNetwork, type)
    assert isinstance(ValueNetwork, type)
    assert isinstance(Config, type)



def test_smoke_train():
    from cartpole_pg import Config, PolicyNetwork, train, loss_vanilla_pg

    cfg = Config(max_episodes=50, eval_every=25, seed=42)
    policy, history = train(loss_vanilla_pg, cfg, method_name="_test_vpg")

    assert isinstance(policy, PolicyNetwork), \
        f"Expected PolicyNetwork, got {type(policy)}"

    for key in ["episode_rewards", "eval_rewards", "policy_losses",
                "grad_variances", "gradient_norms"]:
        assert key in history, f"Missing history key: {key}"

    assert len(history["episode_rewards"]) > 0
    assert len(history["grad_variances"]) > 0

    test_path = os.path.join(RESULTS_DIR, "_test_vpg_seed42.json")
    if os.path.exists(test_path):
        os.remove(test_path)



def test_loss_functions():
    import gymnasium as gym
    from cartpole_pg import (
        PolicyNetwork, ValueNetwork, Config,
        collect_trajectory, loss_vanilla_pg,
        loss_pg_avg_baseline, loss_pg_value_baseline, loss_rloo,
    )

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    cfg = Config()

    loss_fns = [
        ("loss_vanilla_pg", loss_vanilla_pg, False),
        ("loss_pg_avg_baseline", loss_pg_avg_baseline, False),
        ("loss_pg_value_baseline", loss_pg_value_baseline, True),
        ("loss_rloo", loss_rloo, False),
    ]

    for name, loss_fn, needs_value_net in loss_fns:
        policy = PolicyNetwork(state_dim, action_dim, cfg.hidden_dims)
        value_net = ValueNetwork(state_dim, cfg.hidden_dims) if needs_value_net else None

        batch = [collect_trajectory(env, policy) for _ in range(2)]
        loss, info = loss_fn(policy, batch, cfg, value_net=value_net)

        assert isinstance(loss, torch.Tensor), \
            f"{name}: loss is not a Tensor"
        assert loss.requires_grad, \
            f"{name}: loss.requires_grad is False"

        policy.zero_grad()
        if value_net is not None:
            value_net.zero_grad()
        loss.backward()

        max_grad = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.abs().max().item())
        assert max_grad > 1e-10, \
            f"{name}: max gradient norm is {max_grad} (too small)"

        assert isinstance(info, dict), \
            f"{name}: info is not a dict, got {type(info)}"

    env.close()



def test_bc():
    from cartpole_pg import Config, train_bc

    cfg = Config(bc_epochs=2, bc_batch_size=32)
    states = np.random.randn(100, 4).astype(np.float32)
    actions = np.random.randint(0, 2, size=100)

    bc_policy, bc_hist = train_bc(states, actions, cfg)

    for key in ["train_loss", "val_loss", "val_accuracy"]:
        assert key in bc_hist, f"BC history missing key: {key}"

    test_state = torch.FloatTensor(np.random.randn(1, 4))
    dist = bc_policy(test_state)
    probs = dist.probs.detach().numpy().flatten()

    assert len(probs) == 2, f"Expected 2 action probs, got {len(probs)}"
    assert abs(probs.sum() - 1.0) < 1e-5, \
        f"Probabilities don't sum to 1: {probs.sum()}"
    assert all(p >= 0 for p in probs), \
        f"Negative probabilities: {probs}"



def test_results_integrity():
    json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    assert len(json_files) >= 10, \
        f"Expected >=10 JSON files, found {len(json_files)}"

    required_top = ["config", "history"]
    required_history = ["episode_rewards", "eval_rewards", "policy_losses"]
    checked = 0
    for fname in json_files:
        if fname == "bc_results.json":
            continue
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        if "history" not in data:
            continue
        for key in required_top:
            assert key in data, f"{fname} missing top-level key: {key}"
        for key in required_history:
            assert key in data["history"], \
                f"{fname} missing history key: {key}"
        checked += 1

    assert checked >= 10, f"Only {checked} valid training JSONs found"

    assert os.path.exists(SUMMARY_PATH), "summary.json not found"
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    for method, stats in summary.items():
        assert stats["mean_reward"] >= 50, \
            f"{method} has suspiciously low mean_reward: {stats['mean_reward']}"



def test_figures_exist():
    required_figs = [
        "fig1_learning_curves.png",
        "fig2_grad_variance.png",
        "fig3_entropy.png",
        "fig4_rloo_ablation.png",
        "fig5_bc_dataset_size.png",
        "fig6_bc_state_dist.png",
        "fig7_bc_failure_episode.png",
        "fig8_summary_table.png",
    ]

    for fname in required_figs:
        path = os.path.join(FIGURES_DIR, fname)
        assert os.path.exists(path), f"Missing figure: {path}"
        size = os.path.getsize(path)
        assert size > 10_000, \
            f"{fname} is only {size} bytes (expected >10KB)"



def test_report():
    assert os.path.exists(REPORT_PATH), "report/report.pdf not found"

    size = os.path.getsize(REPORT_PATH)
    assert size > 10_000, \
        f"report.pdf is only {size} bytes (expected >10KB)"



def main():
    print("=" * 60)
    print("HW1 Test Suite")
    print("=" * 60)

    tests = [
        ("1. Import test", test_imports),
        ("2. Smoke test (train VPG 50 eps)", test_smoke_train),
        ("3. Loss function tests", test_loss_functions),
        ("4. BC test", test_bc),
        ("5. Results integrity", test_results_integrity),
        ("6. Figures exist", test_figures_exist),
        ("7. Report test", test_report),
    ]

    for name, fn in tests:
        print(f"\n{name}")
        run_test(name, fn)

    print("\n" + "=" * 60)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print("=" * 60)

    if failed:
        print("\nFAILURES:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
