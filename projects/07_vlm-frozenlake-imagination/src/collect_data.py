"""collect FrozenLake transition data for later SFT"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from agents import choose_gt_greedy_action
from frozenlake_utils import (
    ACTION_NAMES,
    deterministic_transition,
    extract_gt_state,
    generate_reachable_random_map,
    make_frozenlake_env,
    render_rgb_array,
)

RESULTS_DIR = Path("results/A3_dataset_collection")
OUTCOMES = {"safe", "hole", "goal", "wall"}
TARGET_RE = re.compile(
    r"^<prediction>Position: \((\d+), (\d+)\)\. Outcome: (safe|hole|goal|wall)</prediction>$"
)


def position_to_list(position: tuple[int, int]) -> list[int]:
    return [int(position[0]), int(position[1])]


def format_position(position: tuple[int, int]) -> str:
    return f"({position[0]}, {position[1]})"


def format_target(next_position: tuple[int, int], outcome: str) -> str:
    return f"<prediction>Position: {format_position(next_position)}. Outcome: {outcome}</prediction>"


def parse_target_text(target_text: str) -> tuple[tuple[int, int], str]:
    match = TARGET_RE.fullmatch(target_text)
    if not match:
        raise ValueError(f"target text does not match required format: {target_text!r}")
    return (int(match.group(1)), int(match.group(2))), match.group(3)


def format_gt_state_text(
    current_position: tuple[int, int],
    goal_position: tuple[int, int],
    hole_positions: Iterable[tuple[int, int]],
) -> str:
    holes = ", ".join(format_position(position) for position in hole_positions)
    return (
        "State: "
        f"player_position={format_position(current_position)}; "
        f"goal_position={format_position(goal_position)}; "
        f"hole_positions=[{holes}]"
    )


def action_name(action_id: int) -> str:
    name = ACTION_NAMES[action_id]
    return name.capitalize()


def choose_action(
    policy_source: str,
    rng: np.random.Generator,
    desc: list[str],
    observation: int,
) -> int:
    state = extract_gt_state(desc, observation)
    if policy_source == "random":
        return int(rng.integers(0, 4))
    if policy_source == "greedy":
        decision = choose_gt_greedy_action(state.grid, state.player_position, state.goal_position)
        if decision is None:
            return int(rng.integers(0, 4))
        return decision.action
    raise ValueError(f"unknown policy source: {policy_source}")


def gym_outcome(reward: float, terminated: bool, predicted_outcome: str) -> str:
    if terminated and reward > 0:
        return "goal"
    if terminated:
        return "hole"
    return predicted_outcome


def collect_for_policy(
    map_seed: int,
    split: str,
    desc: list[str],
    policy_source: str,
    transitions_per_policy_map: int,
    results_dir: Path,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    env = make_frozenlake_env(desc, seed=map_seed)
    observation, info = env.reset(seed=map_seed)
    del info

    transitions: list[dict] = []
    verification_rows: list[dict] = []
    episode_id = 0
    episode_step = 0

    image_dir = results_dir / "images" / split / f"seed_{map_seed:04d}" / policy_source
    image_dir.mkdir(parents=True, exist_ok=True)

    for transition_index in range(transitions_per_policy_map):
        state = extract_gt_state(desc, observation)
        action_id = choose_action(policy_source, rng, desc, observation)
        frame = render_rgb_array(env)
        image_path = image_dir / f"ep_{episode_id:03d}_step_{episode_step:03d}.png"
        Image.fromarray(frame).save(image_path)

        predicted_next_position, predicted_outcome = deterministic_transition(
            state.grid,
            state.player_position,
            action_id,
        )
        next_observation, reward, terminated, truncated, info = env.step(action_id)
        del info
        next_state = extract_gt_state(desc, next_observation)
        actual_next_position = next_state.player_position
        actual_outcome = gym_outcome(float(reward), bool(terminated), predicted_outcome)

        if actual_next_position != predicted_next_position:
            raise AssertionError(
                f"position mismatch seed={map_seed} policy={policy_source} "
                f"action={action_id}: predicted={predicted_next_position} actual={actual_next_position}"
            )
        if actual_outcome != predicted_outcome:
            raise AssertionError(
                f"outcome mismatch seed={map_seed} policy={policy_source} "
                f"action={action_id}: predicted={predicted_outcome} actual={actual_outcome}"
            )

        target_text = format_target(predicted_next_position, predicted_outcome)
        parsed_position, parsed_outcome = parse_target_text(target_text)
        if parsed_position != predicted_next_position or parsed_outcome != predicted_outcome:
            raise AssertionError(f"target parse mismatch: {target_text}")

        transition_id = f"{split}_seed{map_seed:04d}_{policy_source}_t{transition_index:03d}"
        row = {
            "transition_id": transition_id,
            "map_seed": map_seed,
            "split": split,
            "policy_source": policy_source,
            "episode_id": episode_id,
            "step_index": episode_step,
            "transition_index": transition_index,
            "image_path": str(image_path),
            "action_id": action_id,
            "action_name": action_name(action_id),
            "current_position": position_to_list(state.player_position),
            "goal_position": position_to_list(state.goal_position),
            "hole_positions": [position_to_list(position) for position in state.hole_positions],
            "next_position": position_to_list(predicted_next_position),
            "outcome": predicted_outcome,
            "target_text": target_text,
            "gt_state_text": format_gt_state_text(
                state.player_position,
                state.goal_position,
                state.hole_positions,
            ),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "map_text": desc,
        }
        transitions.append(row)

        if len(verification_rows) < 5:
            verification_rows.append(
                {
                    "transition_id": transition_id,
                    "predicted_next_position": position_to_list(predicted_next_position),
                    "actual_next_position": position_to_list(actual_next_position),
                    "predicted_outcome": predicted_outcome,
                    "actual_outcome": actual_outcome,
                    "target_parse_position": position_to_list(parsed_position),
                    "target_parse_outcome": parsed_outcome,
                }
            )

        observation = next_observation
        episode_step += 1
        if terminated or truncated:
            episode_id += 1
            episode_step = 0
            observation, info = env.reset(seed=map_seed + episode_id)
            del info

    env.close()
    return transitions, verification_rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_sample_files(results_dir: Path, transitions: list[dict], count: int = 5) -> list[dict]:
    sample_dir = results_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = transitions[:count]
    write_jsonl(sample_dir / "sample_rows.jsonl", samples)
    for index, row in enumerate(samples):
        text = [
            f"transition_id: {row['transition_id']}",
            f"rendered_image_path: {row['image_path']}",
            "map:",
            *row["map_text"],
            f"action: {row['action_name']} ({row['action_id']})",
            f"gt_state_text: {row['gt_state_text']}",
            f"target_text: {row['target_text']}",
            "",
        ]
        (sample_dir / f"sample_{index:02d}.txt").write_text("\n".join(text), encoding="utf-8")
    return samples


def summarize(transitions: list[dict], map_rows: list[dict], verification_rows: list[dict]) -> dict:
    split_counts = Counter(row["split"] for row in transitions)
    policy_counts = Counter((row["split"], row["policy_source"]) for row in transitions)
    outcome_counts = Counter((row["split"], row["outcome"]) for row in transitions)
    map_split_counts = Counter(row["split"] for row in map_rows)

    return {
        "num_maps": len(map_rows),
        "train_map_count": map_split_counts["train"],
        "val_map_count": map_split_counts["val"],
        "total_transitions": len(transitions),
        "target_range": "about 2000-5000 transitions",
        "transitions_per_policy_map": len(transitions) // (len(map_rows) * 2) if map_rows else 0,
        "transition_counts_by_split": dict(split_counts),
        "transition_counts_by_split_and_policy": {
            f"{split}_{policy}": count for (split, policy), count in sorted(policy_counts.items())
        },
        "outcome_distribution_by_split": {
            f"{split}_{outcome}": count for (split, outcome), count in sorted(outcome_counts.items())
        },
        "label_verification_sample": verification_rows[:10],
    }


def validate_split_integrity(map_rows: list[dict], transitions: list[dict]) -> None:
    seed_to_split = {row["map_seed"]: row["split"] for row in map_rows}
    seen_transition_splits: dict[int, set[str]] = defaultdict(set)
    for row in transitions:
        expected_split = seed_to_split[row["map_seed"]]
        if row["split"] != expected_split:
            raise AssertionError(
                f"transition split mismatch for seed {row['map_seed']}: "
                f"{row['split']} != {expected_split}"
            )
        seen_transition_splits[row["map_seed"]].add(row["split"])

    bad = {seed: splits for seed, splits in seen_transition_splits.items() if len(splits) != 1}
    if bad:
        raise AssertionError(f"map seeds appear in multiple splits: {bad}")


def write_csv_summary(path: Path, transitions: list[dict]) -> None:
    fieldnames = [
        "transition_id",
        "map_seed",
        "split",
        "policy_source",
        "action_name",
        "current_position",
        "next_position",
        "outcome",
        "image_path",
        "target_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in transitions:
            writer.writerow({field: row[field] for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-maps", type=int, default=100)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--transitions-per-policy-map", type=int, default=15)
    parser.add_argument("--map-seed-start", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    train_count = int(args.num_maps * args.train_frac)
    map_seeds = list(range(args.map_seed_start, args.map_seed_start + args.num_maps))
    rng = np.random.default_rng(20260520)

    all_transitions: list[dict] = []
    map_rows: list[dict] = []
    verification_rows: list[dict] = []

    for map_index, map_seed in enumerate(map_seeds):
        split = "train" if map_index < train_count else "val"
        desc = generate_reachable_random_map(seed=map_seed)
        map_rows.append({"map_seed": map_seed, "split": split, "map_text": desc})
        for policy_source in ("greedy", "random"):
            transitions, checks = collect_for_policy(
                map_seed=map_seed,
                split=split,
                desc=desc,
                policy_source=policy_source,
                transitions_per_policy_map=args.transitions_per_policy_map,
                results_dir=args.results_dir,
                rng=rng,
            )
            all_transitions.extend(transitions)
            verification_rows.extend(checks)

    validate_split_integrity(map_rows, all_transitions)
    for row in all_transitions:
        parsed_position, parsed_outcome = parse_target_text(row["target_text"])
        if position_to_list(parsed_position) != row["next_position"] or parsed_outcome != row["outcome"]:
            raise AssertionError(f"target row parse failed: {row['transition_id']}")
        if row["outcome"] not in OUTCOMES:
            raise AssertionError(f"invalid outcome {row['outcome']} in {row['transition_id']}")

    train_rows = [row for row in all_transitions if row["split"] == "train"]
    val_rows = [row for row in all_transitions if row["split"] == "val"]
    sample_rows = write_sample_files(args.results_dir, all_transitions)
    summary = summarize(all_transitions, map_rows, verification_rows)
    summary["sample_transition_ids"] = [row["transition_id"] for row in sample_rows]

    write_jsonl(args.results_dir / "transitions.jsonl", all_transitions)
    write_jsonl(args.results_dir / "train.jsonl", train_rows)
    write_jsonl(args.results_dir / "val.jsonl", val_rows)
    write_jsonl(args.results_dir / "maps.jsonl", map_rows)
    write_csv_summary(args.results_dir / "summary.csv", all_transitions)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage A3 dataset collection")
    print(f"results_dir={args.results_dir}")
    print(f"num_maps={summary['num_maps']}")
    print(f"train_map_count={summary['train_map_count']} val_map_count={summary['val_map_count']}")
    print(f"total_transitions={summary['total_transitions']}")
    print(f"transition_counts_by_split={summary['transition_counts_by_split']}")
    print(f"transition_counts_by_split_and_policy={summary['transition_counts_by_split_and_policy']}")
    print(f"outcome_distribution_by_split={summary['outcome_distribution_by_split']}")
    print("split_integrity=passed")
    print("target_parse_check=passed")
    print("gym_transition_label_check=passed")
    print(f"summary_json={args.results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
