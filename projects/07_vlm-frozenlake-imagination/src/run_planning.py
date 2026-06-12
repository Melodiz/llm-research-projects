"""Stage A6: planning comparison across FrozenLake conditions"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from agents import GREEDY_TIE_BREAK_ORDER, manhattan_distance
from collect_data import format_gt_state_text
from eval_world_model import parse_prediction
from frozenlake_utils import (
    ACTION_NAMES,
    deterministic_transition,
    extract_gt_state,
    generate_reachable_random_map,
    make_frozenlake_env,
    render_rgb_array,
)
from run_a2_reactive_vlm import (
    FALLBACK_ACTION_ID,
    FALLBACK_ACTION_NAME,
    QwenVLReactivePolicy,
    parse_action_response,
)
from train_sft import INSTRUCTION, SYSTEM_PROMPT, Condition

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_RESULTS_DIR = Path("results/A6_planning_comparison")
DEFAULT_ADAPTER_ROOT = Path("results/A4_sft_training")
DEFAULT_SEEDS = list(range(100, 130))
CONDITIONS = (
    "reactive_vlm",
    "lookahead_zeroshot_vlm",
    "lookahead_sft_image_text",
    "lookahead_sft_image_only",
    "lookahead_gt",
)
MODEL_CONDITIONS = {
    "reactive_vlm",
    "lookahead_zeroshot_vlm",
    "lookahead_sft_image_text",
    "lookahead_sft_image_only",
}
LOOKAHEAD_CONDITIONS = {
    "lookahead_zeroshot_vlm",
    "lookahead_sft_image_text",
    "lookahead_sft_image_only",
    "lookahead_gt",
}

ACTION_ID_TO_NAME = {action_id: name.capitalize() for action_id, name in ACTION_NAMES.items()}
ACTION_NAME_TO_ID = {name: action_id for action_id, name in ACTION_ID_TO_NAME.items()}
LOOKAHEAD_TIE_BREAK_ORDER = GREEDY_TIE_BREAK_ORDER

ZERO_SHOT_WORLD_SYSTEM_PROMPT = (
    "You are a FrozenLake transition predictor. Given the rendered image and one action, "
    "predict the player's next position and transition outcome. Return only the required "
    "prediction tag."
)

ZERO_SHOT_WORLD_USER_TEMPLATE = (
    "Predict the result of taking this action in deterministic 8x8 FrozenLake.\n"
    "Use zero-indexed (row, col) coordinates from the top-left.\n"
    "Visible cracked/dark tiles are holes; safe frozen tiles can be crossed; the goal is the target tile.\n"
    "Action: {action_name}\n"
    "Answer exactly in this format:\n"
    "<prediction>Position: (r, c). Outcome: safe</prediction>"
)


@dataclass
class Prediction:
    action_id: int
    action_name: str
    raw_response: str
    parse_compliant: bool
    predicted_position: list[int] | None
    predicted_outcome: str | None
    parse_error: str | None


class WorldPredictor(Protocol):
    name: str
    model_id: str

    def predict(
        self,
        frame: np.ndarray,
        action_id: int,
        gt_state_text: str,
        desc: list[str],
        current_position: tuple[int, int],
    ) -> Prediction:
        """Predict one-step next position/outcome for one action."""


class GTWorldPredictor:
    name = "gt"
    model_id = "ground_truth_deterministic_transition"

    def predict(
        self,
        frame: np.ndarray,
        action_id: int,
        gt_state_text: str,
        desc: list[str],
        current_position: tuple[int, int],
    ) -> Prediction:
        del frame, gt_state_text
        next_position, outcome = deterministic_transition(desc, current_position, action_id)
        raw = f"<prediction>Position: ({next_position[0]}, {next_position[1]}). Outcome: {outcome}</prediction>"
        return Prediction(
            action_id=action_id,
            action_name=ACTION_ID_TO_NAME[action_id],
            raw_response=raw,
            parse_compliant=True,
            predicted_position=[next_position[0], next_position[1]],
            predicted_outcome=outcome,
            parse_error=None,
        )


class FakeWorldPredictor:
    """Local smoke predictor. It uses GT labels but exercises parser paths."""

    name = "fake_gt_text"
    model_id = "fake_gt_text"

    def predict(
        self,
        frame: np.ndarray,
        action_id: int,
        gt_state_text: str,
        desc: list[str],
        current_position: tuple[int, int],
    ) -> Prediction:
        del frame, gt_state_text
        next_position, outcome = deterministic_transition(desc, current_position, action_id)
        raw = f"<prediction>Position: ({next_position[0]}, {next_position[1]}). Outcome: {outcome}</prediction>"
        return parse_world_prediction(raw, action_id)


class QwenWorldPredictor:
    def __init__(
        self,
        model_id: str,
        condition: str,
        adapter_root: Path,
        torch_dtype: str,
        device_map: str,
        max_new_tokens: int,
    ) -> None:
        self.name = condition
        self.model_id = model_id
        self.condition = condition
        self.max_new_tokens = max_new_tokens

        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype: str | torch.dtype = torch_dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.process_vision_info = process_vision_info
        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )

        if condition in {"lookahead_sft_image_text", "lookahead_sft_image_only"}:
            adapter_name = "image_text" if condition == "lookahead_sft_image_text" else "image_only"
            self.model = PeftModel.from_pretrained(self.model, adapter_root / adapter_name / "adapter")
        self.model.eval()

    def predict(
        self,
        frame: np.ndarray,
        action_id: int,
        gt_state_text: str,
        desc: list[str],
        current_position: tuple[int, int],
    ) -> Prediction:
        return self.predict_many(frame, [action_id], gt_state_text, desc, current_position)[0]

    def predict_many(
        self,
        frame: np.ndarray,
        action_ids: list[int],
        gt_state_text: str,
        desc: list[str],
        current_position: tuple[int, int],
    ) -> list[Prediction]:
        del desc, current_position
        image = Image.fromarray(frame)
        messages_batch = []
        for action_id in action_ids:
            action_name = ACTION_ID_TO_NAME[action_id]
            if self.condition == "lookahead_zeroshot_vlm":
                messages = build_zeroshot_world_messages(image, action_name)
            elif self.condition == "lookahead_sft_image_text":
                messages = build_sft_world_messages(image, action_name, "image_text", gt_state_text)
            elif self.condition == "lookahead_sft_image_only":
                messages = build_sft_world_messages(image, action_name, "image_only", gt_state_text)
            else:
                raise ValueError(f"unsupported Qwen world condition {self.condition}")
            messages_batch.append(messages)

        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        image_inputs, video_inputs = self.process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        raw_responses = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            parse_world_prediction(raw.strip(), action_id)
            for raw, action_id in zip(raw_responses, action_ids, strict=True)
        ]


def build_zeroshot_world_messages(image: Image.Image, action_name: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": ZERO_SHOT_WORLD_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ZERO_SHOT_WORLD_USER_TEMPLATE.format(action_name=action_name)},
            ],
        },
    ]


def build_sft_world_messages(
    image: Image.Image,
    action_name: str,
    condition: Condition,
    gt_state_text: str,
) -> list[dict[str, Any]]:
    user_parts = [INSTRUCTION, f"Action: {action_name}"]
    if condition == "image_text":
        user_parts.append(gt_state_text)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "\n".join(user_parts)},
            ],
        },
    ]


def parse_world_prediction(raw: str, action_id: int) -> Prediction:
    parsed = parse_prediction(raw)
    return Prediction(
        action_id=action_id,
        action_name=ACTION_ID_TO_NAME[action_id],
        raw_response=raw,
        parse_compliant=bool(parsed["parse_compliant"]),
        predicted_position=parsed["position"],
        predicted_outcome=parsed["outcome"],
        parse_error=parsed["error"],
    )


def final_outcome_from_step(reward: float, terminated: bool, truncated: bool) -> str | None:
    if terminated and reward > 0:
        return "goal"
    if terminated:
        return "hole"
    if truncated:
        return "truncated"
    return None


def choose_lookahead_action(
    predictions: list[Prediction],
    goal_position: tuple[int, int],
) -> tuple[int, bool, str]:
    candidates = []
    for order_index, action_id in enumerate(LOOKAHEAD_TIE_BREAK_ORDER):
        prediction = next(item for item in predictions if item.action_id == action_id)
        if not prediction.parse_compliant:
            continue
        if prediction.predicted_outcome in {"hole", "wall"}:
            continue
        if prediction.predicted_position is None:
            continue
        distance = manhattan_distance(tuple(prediction.predicted_position), goal_position)
        candidates.append((distance, order_index, action_id))
    if not candidates:
        return FALLBACK_ACTION_ID, True, "no parsed non-hole non-wall candidate"
    _, _, action_id = min(candidates)
    return action_id, False, "min_manhattan_to_goal"


def run_reactive_episode(seed: int, condition_dir: Path, policy: Any, max_steps_override: int | None) -> dict:
    desc = generate_reachable_random_map(seed=seed)
    env = make_frozenlake_env(desc, seed=seed)
    observation, info = env.reset(seed=seed)
    del info
    max_steps = max_steps_override or int(getattr(env.spec, "max_episode_steps", 100) or 100)
    trace = []
    raw_rows = []
    total_reward = 0.0
    fallback_count = 0
    compliant_count = 0
    final_outcome = "step_cap"

    for step in range(max_steps):
        state = extract_gt_state(desc, observation)
        frame = render_rgb_array(env)
        raw = policy.generate_action_text(frame)
        parsed = parse_action_response(raw)
        compliant_count += int(parsed.compliant)
        fallback_count += int(parsed.fallback_used)
        raw_rows.append({"seed": seed, "step": step, "raw_response": raw})

        observation, reward, terminated, truncated, info = env.step(parsed.action_id)
        del info
        total_reward += float(reward)
        next_state = extract_gt_state(desc, observation)
        outcome = final_outcome_from_step(float(reward), bool(terminated), bool(truncated))
        if outcome is not None:
            final_outcome = outcome

        trace.append(
            {
                "step": step,
                "player_position": list(state.player_position),
                "action_id": parsed.action_id,
                "action_name": parsed.action_name,
                "raw_response": raw,
                "format_compliant": parsed.compliant,
                "fallback_used": parsed.fallback_used,
                "next_position": list(next_state.player_position),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    env.close()
    return episode_result(
        seed=seed,
        desc=desc,
        trace=trace,
        raw_rows=raw_rows,
        total_reward=total_reward,
        final_outcome=final_outcome,
        query_count=len(trace),
        compliant_count=compliant_count,
        parse_failure_count=len(trace) - compliant_count,
        fallback_count=fallback_count,
    )


def run_lookahead_episode(
    seed: int,
    condition_dir: Path,
    predictor: WorldPredictor,
    max_steps_override: int | None,
) -> dict:
    desc = generate_reachable_random_map(seed=seed)
    env = make_frozenlake_env(desc, seed=seed)
    observation, info = env.reset(seed=seed)
    del info
    max_steps = max_steps_override or int(getattr(env.spec, "max_episode_steps", 100) or 100)
    trace = []
    raw_rows = []
    total_reward = 0.0
    fallback_count = 0
    query_count = 0
    compliant_count = 0
    parse_failure_count = 0
    final_outcome = "step_cap"

    for step in range(max_steps):
        state = extract_gt_state(desc, observation)
        frame = render_rgb_array(env)
        gt_text = format_gt_state_text(state.player_position, state.goal_position, state.hole_positions)
        action_ids = sorted(ACTION_ID_TO_NAME)
        predict_many = getattr(predictor, "predict_many", None)
        if predict_many is not None:
            predictions = predict_many(frame, action_ids, gt_text, desc, state.player_position)
        else:
            predictions = [
                predictor.predict(frame, action_id, gt_text, desc, state.player_position)
                for action_id in action_ids
            ]
        query_count += len(predictions)
        compliant_count += sum(item.parse_compliant for item in predictions)
        parse_failure_count += sum(not item.parse_compliant for item in predictions)
        for prediction in predictions:
            raw_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "action_id": prediction.action_id,
                    "action_name": prediction.action_name,
                    "raw_response": prediction.raw_response,
                    "parse_compliant": prediction.parse_compliant,
                    "predicted_position": prediction.predicted_position,
                    "predicted_outcome": prediction.predicted_outcome,
                }
            )

        action_id, used_fallback, decision_reason = choose_lookahead_action(predictions, state.goal_position)
        fallback_count += int(used_fallback)
        observation, reward, terminated, truncated, info = env.step(action_id)
        del info
        total_reward += float(reward)
        next_state = extract_gt_state(desc, observation)
        outcome = final_outcome_from_step(float(reward), bool(terminated), bool(truncated))
        if outcome is not None:
            final_outcome = outcome

        trace.append(
            {
                "step": step,
                "player_position": list(state.player_position),
                "goal_position": list(state.goal_position),
                "gt_state_text": gt_text,
                "predictions": [prediction.__dict__ for prediction in predictions],
                "chosen_action_id": action_id,
                "chosen_action_name": ACTION_ID_TO_NAME[action_id],
                "fallback_used": used_fallback,
                "decision_reason": decision_reason,
                "actual_next_position": list(next_state.player_position),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    env.close()
    return episode_result(
        seed=seed,
        desc=desc,
        trace=trace,
        raw_rows=raw_rows,
        total_reward=total_reward,
        final_outcome=final_outcome,
        query_count=query_count,
        compliant_count=compliant_count,
        parse_failure_count=parse_failure_count,
        fallback_count=fallback_count,
    )


def episode_result(
    seed: int,
    desc: list[str],
    trace: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    total_reward: float,
    final_outcome: str,
    query_count: int,
    compliant_count: int,
    parse_failure_count: int,
    fallback_count: int,
) -> dict:
    return {
        "seed": seed,
        "map": desc,
        "steps": len(trace),
        "total_reward": total_reward,
        "success": bool(final_outcome == "goal"),
        "final_outcome": final_outcome,
        "hole": bool(final_outcome == "hole"),
        "truncated": bool(final_outcome == "truncated"),
        "query_count": query_count,
        "format_compliant_count": compliant_count,
        "parse_failure_count": parse_failure_count,
        "fallback_count": fallback_count,
        "format_compliance_rate": compliant_count / query_count if query_count else 1.0,
        "parse_failure_rate": parse_failure_count / query_count if query_count else 0.0,
        "fallback_action_rate": fallback_count / len(trace) if trace else 0.0,
        "trace": trace,
        "raw_model_rows": raw_rows,
    }


def summarize_episodes(condition: str, episodes: list[dict[str, Any]], bootstrap_seed: int) -> dict:
    count = len(episodes)
    success_rate = sum(ep["success"] for ep in episodes) / count if count else 0.0
    hole_rate = sum(ep["final_outcome"] == "hole" for ep in episodes) / count if count else 0.0
    truncation_rate = sum(ep["final_outcome"] == "truncated" for ep in episodes) / count if count else 0.0
    mean_steps = sum(ep["steps"] for ep in episodes) / count if count else 0.0
    total_queries = sum(ep["query_count"] for ep in episodes)
    compliant = sum(ep["format_compliant_count"] for ep in episodes)
    parse_failures = sum(ep["parse_failure_count"] for ep in episodes)
    fallbacks = sum(ep["fallback_count"] for ep in episodes)
    total_steps = sum(ep["steps"] for ep in episodes)
    return {
        "condition": condition,
        "episodes": count,
        "seeds": [ep["seed"] for ep in episodes],
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "hole_rate": hole_rate,
        "truncation_rate": truncation_rate,
        "format_compliance_rate": compliant / total_queries if total_queries else 1.0,
        "prediction_parse_failure_rate": parse_failures / total_queries if total_queries else 0.0,
        "fallback_action_rate": fallbacks / total_steps if total_steps else 0.0,
        "success_count": sum(ep["success"] for ep in episodes),
        "hole_count": sum(ep["final_outcome"] == "hole" for ep in episodes),
        "truncated_count": sum(ep["final_outcome"] == "truncated" for ep in episodes),
        "total_steps": total_steps,
        "total_queries": total_queries,
        "fallback_count": fallbacks,
        "bootstrap_ci": bootstrap_ci(episodes, bootstrap_seed),
    }


def bootstrap_ci(episodes: list[dict[str, Any]], seed: int, samples: int = 2000) -> dict[str, dict[str, float]]:
    if not episodes:
        return {}
    rng = np.random.default_rng(seed)
    n = len(episodes)
    metrics = {"success_rate": [], "hole_rate": [], "mean_steps": []}
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        subset = [episodes[i] for i in idx]
        metrics["success_rate"].append(sum(ep["success"] for ep in subset) / n)
        metrics["hole_rate"].append(sum(ep["final_outcome"] == "hole" for ep in subset) / n)
        metrics["mean_steps"].append(sum(ep["steps"] for ep in subset) / n)
    return {
        key: {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
        for key, values in metrics.items()
    }


def write_condition_outputs(condition_dir: Path, episodes: list[dict[str, Any]], summary: dict) -> None:
    condition_dir.mkdir(parents=True, exist_ok=True)
    write_json(condition_dir / "summary.json", summary)
    with (condition_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "success",
            "final_outcome",
            "steps",
            "total_reward",
            "format_compliance_rate",
            "parse_failure_rate",
            "fallback_action_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            writer.writerow({field: ep[field] for field in fieldnames})

    with (condition_dir / "raw_model_responses.jsonl").open("w", encoding="utf-8") as handle:
        for ep in episodes:
            for row in ep["raw_model_rows"]:
                handle.write(json.dumps(row) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def condition_output_name(condition: str) -> str:
    return condition


def make_reactive_policy(args: argparse.Namespace) -> Any:
    if args.backend == "fake":
        from run_a2_reactive_vlm import FakeReactivePolicy

        return FakeReactivePolicy()
    return QwenVLReactivePolicy(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )


def make_world_predictor(args: argparse.Namespace, condition: str) -> WorldPredictor:
    if condition == "lookahead_gt":
        return GTWorldPredictor()
    if args.backend == "fake":
        return FakeWorldPredictor()
    return QwenWorldPredictor(
        model_id=args.model_id,
        condition=condition,
        adapter_root=args.adapter_root,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )


def save_prompt_files(results_dir: Path) -> None:
    prompt_dir = results_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "zeroshot_reactive_system.txt").write_text(
        "See run_a2_reactive_vlm.SYSTEM_PROMPT\n",
        encoding="utf-8",
    )
    (prompt_dir / "zeroshot_world_system.txt").write_text(ZERO_SHOT_WORLD_SYSTEM_PROMPT + "\n", encoding="utf-8")
    (prompt_dir / "zeroshot_world_user_template.txt").write_text(
        ZERO_SHOT_WORLD_USER_TEMPLATE + "\n",
        encoding="utf-8",
    )
    (prompt_dir / "sft_system.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (prompt_dir / "sft_instruction.txt").write_text(INSTRUCTION + "\n", encoding="utf-8")


def run_condition(args: argparse.Namespace, condition: str) -> dict:
    condition_dir = args.results_dir / condition_output_name(condition)
    condition_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    start = time.time()

    if condition == "reactive_vlm":
        policy = make_reactive_policy(args)
        model_id = getattr(policy, "model_id", args.model_id)
        for seed in args.seeds:
            ep = run_reactive_episode(seed, condition_dir, policy, args.max_episode_steps)
            episodes.append(ep)
            write_json(condition_dir / f"seed_{seed}_trace.json", ep)
            print(f"{condition} seed={seed} outcome={ep['final_outcome']} steps={ep['steps']}", flush=True)
    else:
        predictor = make_world_predictor(args, condition)
        model_id = predictor.model_id
        for seed in args.seeds:
            ep = run_lookahead_episode(seed, condition_dir, predictor, args.max_episode_steps)
            episodes.append(ep)
            write_json(condition_dir / f"seed_{seed}_trace.json", ep)
            print(f"{condition} seed={seed} outcome={ep['final_outcome']} steps={ep['steps']}", flush=True)

    summary = summarize_episodes(condition, episodes, args.bootstrap_seed)
    summary["runtime_seconds"] = time.time() - start
    summary["model_id"] = model_id
    summary["backend"] = args.backend
    write_condition_outputs(condition_dir, episodes, summary)
    return summary


def write_comparison(results_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        raise ValueError("no summaries available for comparison")
    rows = []
    for summary in summaries:
        rows.append(
            {
                "condition": summary["condition"],
                "episodes": summary["episodes"],
                "success_rate": summary["success_rate"],
                "mean_steps": summary["mean_steps"],
                "hole_rate": summary["hole_rate"],
                "truncation_rate": summary["truncation_rate"],
                "format_compliance_rate": summary["format_compliance_rate"],
                "prediction_parse_failure_rate": summary["prediction_parse_failure_rate"],
                "fallback_action_rate": summary["fallback_action_rate"],
                "runtime_seconds": summary["runtime_seconds"],
            }
        )
    with (results_dir / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(results_dir / "bootstrap_ci.json", {row["condition"]: row["bootstrap_ci"] for row in summaries})


def aggregate_existing_results(args: argparse.Namespace) -> None:
    summaries = []
    seed_sets = {}
    for condition in CONDITIONS:
        summary_path = args.results_dir / condition / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing summary for {condition}: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        seed_sets[condition] = summary["seeds"]

    reference = seed_sets[CONDITIONS[0]]
    mismatched = {condition: seeds for condition, seeds in seed_sets.items() if seeds != reference}
    if mismatched:
        raise AssertionError(f"conditions do not share identical seed list: {mismatched}")

    write_comparison(args.results_dir, summaries)
    print("Stage A6 aggregation complete")
    print(f"conditions={list(CONDITIONS)}")
    print(f"shared_seeds={reference}")
    print(f"comparison_table={args.results_dir / 'comparison_table.csv'}")
    print(f"bootstrap_ci={args.results_dir / 'bootstrap_ci.json'}")


def run_parser_smoke() -> None:
    cases = [
        "<prediction>Position: (1, 2). Outcome: safe</prediction>",
        "<prediction>Position: (0, 0). Outcome: wall</prediction>",
        "Position: (1, 2). Outcome: safe",
    ]
    print("Stage A6 parser smoke")
    for raw in cases:
        parsed = parse_world_prediction(raw, 1)
        print(
            f"raw={raw!r} compliant={parsed.parse_compliant} "
            f"position={parsed.predicted_position} outcome={parsed.predicted_outcome}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parser-smoke", "run", "aggregate"], default="run")
    parser.add_argument("--condition", choices=[*CONDITIONS, "all"], default="lookahead_gt")
    parser.add_argument("--backend", choices=["qwen", "fake"], default="qwen")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=20260521)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "parser-smoke":
        run_parser_smoke()
        return
    if args.mode == "aggregate":
        aggregate_existing_results(args)
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_prompt_files(args.results_dir)
    selected = list(CONDITIONS) if args.condition == "all" else [args.condition]
    write_json(
        args.results_dir / "run_config.json",
        {
            "conditions": selected,
            "seeds": args.seeds,
            "seed_rationale": "Held-out seeds 100..129 avoid A3 train seeds 0..79 and val seeds 80..99.",
            "fallback_action": FALLBACK_ACTION_NAME,
            "fallback_action_id": FALLBACK_ACTION_ID,
            "lookahead_tie_break_order": [
                {"action_id": action_id, "action_name": ACTION_ID_TO_NAME[action_id]}
                for action_id in LOOKAHEAD_TIE_BREAK_ORDER
            ],
            "model_id": args.model_id,
            "backend": args.backend,
            "torch_dtype": args.torch_dtype,
            "max_new_tokens": args.max_new_tokens,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )

    summaries = [run_condition(args, condition) for condition in selected]
    if len(summaries) > 1:
        write_comparison(args.results_dir, summaries)
    print("Stage A6 planning comparison complete")
    for summary in summaries:
        print(
            f"{summary['condition']}: success={summary['success_rate']:.3f} "
            f"hole={summary['hole_rate']:.3f} trunc={summary['truncation_rate']:.3f} "
            f"mean_steps={summary['mean_steps']:.2f}"
        )


if __name__ == "__main__":
    main()
