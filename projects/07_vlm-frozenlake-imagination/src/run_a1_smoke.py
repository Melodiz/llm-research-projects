"""Smoke tests for Stage A1 FrozenLake GT environment and greedy planner"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import GREEDY_TIE_BREAK_ORDER, choose_gt_greedy_action
from frozenlake_utils import (
    ACTION_NAMES,
    extract_gt_state,
    generate_reachable_random_map,
    is_bfs_reachable,
    make_frozenlake_env,
    render_rgb_array,
)

DEFAULT_RESULTS_DIR = Path("results/A1_env_gt_planner")


def run_seed(seed: int, results_dir: Path, max_steps: int = 128) -> dict:
    desc = generate_reachable_random_map(seed=seed)
    env = make_frozenlake_env(desc, seed=seed)
    observation, info = env.reset(seed=seed)
    del info

    first_frame = render_rgb_array(env)
    frame_path = results_dir / f"seed_{seed}_initial_rgb.npy"
    map_path = results_dir / f"seed_{seed}_map.txt"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(desc) + "\n")

    # Store a small render artifact without adding image dependencies.
    import numpy as np

    np.save(frame_path, first_frame)

    trace = []
    terminated = False
    truncated = False
    total_reward = 0.0

    for step in range(max_steps):
        state = extract_gt_state(desc, observation)
        decision = choose_gt_greedy_action(state.grid, state.player_position, state.goal_position)
        if decision is None:
            trace.append(
                {
                    "step": step,
                    "position": state.player_position,
                    "decision": None,
                    "reason": "no safe non-wall action available",
                }
            )
            break

        predicted_position = decision.next_position
        predicted_outcome = decision.outcome
        observation, reward, terminated, truncated, info = env.step(decision.action)
        del info
        actual_state = extract_gt_state(desc, observation)
        total_reward += float(reward)
        trace.append(
            {
                "step": step,
                "position": state.player_position,
                "action": decision.action_name,
                "predicted_position": predicted_position,
                "predicted_outcome": predicted_outcome,
                "actual_position": actual_state.player_position,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if actual_state.player_position != predicted_position:
            raise AssertionError(
                f"seed {seed} transition mismatch: predicted {predicted_position}, "
                f"actual {actual_state.player_position}"
            )
        if terminated or truncated:
            break

    env.close()

    final_state = extract_gt_state(desc, observation)
    result = {
        "seed": seed,
        "reachable": is_bfs_reachable(desc),
        "map": desc,
        "holes": list(final_state.hole_positions),
        "goal": final_state.goal_position,
        "final_position": final_state.player_position,
        "steps": len(trace),
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(terminated and total_reward > 0),
        "trace": trace,
        "render_artifact": str(frame_path),
        "map_artifact": str(map_path),
    }

    with (results_dir / f"seed_{seed}_trace.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results = [run_seed(seed, args.results_dir) for seed in args.seeds]
    summary_path = args.results_dir / "smoke_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Stage A1 FrozenLake smoke")
    print(f"results_dir={args.results_dir}")
    print(
        "tie_break_order="
        + ",".join(f"{action}:{ACTION_NAMES[action]}" for action in GREEDY_TIE_BREAK_ORDER)
    )
    for result in results:
        print(
            f"seed={result['seed']} reachable={result['reachable']} "
            f"success={result['success']} steps={result['steps']} "
            f"reward={result['total_reward']} final={tuple(result['final_position'])}"
        )
        print("map:")
        for row in result["map"]:
            print(row)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
