"""Stage A2: zero-shot reactive VLM baseline for FrozenLake"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from frozenlake_utils import (
    ACTION_NAMES,
    extract_gt_state,
    generate_reachable_random_map,
    make_frozenlake_env,
    render_rgb_array,
)

DEFAULT_RESULTS_DIR = Path("results/A2_reactive_vlm")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_SEEDS = list(range(30))
FALLBACK_ACTION_NAME = "Right"
FALLBACK_ACTION_ID = 2
ACTION_NAME_TO_ID = {"Left": 0, "Down": 1, "Right": 2, "Up": 3}

SYSTEM_PROMPT = """You are controlling the player in an 8x8 FrozenLake game from a rendered image.
Choose exactly one action from: Left, Down, Right, Up.
Return only the required XML-like answer tag and no other text."""

USER_PROMPT = """Look at the current FrozenLake image and choose the next move.
The visible symbols are:
- the player marks the current location;
- cracked/dark tiles are holes and should be avoided;
- the goal is the target tile;
- safe frozen tiles can be crossed.

Choose exactly one action from: Left, Down, Right, Up.
Answer in exactly this format:
<answer>Down</answer>"""


@dataclass(frozen=True)
class ParsedAction:
    raw_response: str
    compliant: bool
    action_name: str
    action_id: int
    fallback_used: bool
    error: str | None = None


def parse_action_response(raw_response: str) -> ParsedAction:
    """Parse `<answer>Action</answer>` and map it to a Gymnasium action ID.

    Parsing is intentionally strict: the full response, after leading/trailing
    whitespace is removed, must be exactly one answer tag containing one of
    `Left`, `Down`, `Right`, or `Up`. If parsing fails, the deterministic
    fallback action is `Right` / action ID 2, and the response is non-compliant.
    """

    text = raw_response.strip()
    match = re.fullmatch(r"<answer>(Left|Down|Right|Up)</answer>", text)
    if not match:
        return ParsedAction(
            raw_response=raw_response,
            compliant=False,
            action_name=FALLBACK_ACTION_NAME,
            action_id=FALLBACK_ACTION_ID,
            fallback_used=True,
            error="expected exactly <answer>{Left|Down|Right|Up}</answer>",
        )

    action_name = match.group(1)
    return ParsedAction(
        raw_response=raw_response,
        compliant=True,
        action_name=action_name,
        action_id=ACTION_NAME_TO_ID[action_name],
        fallback_used=False,
    )


class ReactivePolicy(Protocol):
    backend_name: str
    model_id: str

    def generate_action_text(self, frame: np.ndarray) -> str:
        """Return raw model text for one rendered RGB frame."""


class FakeReactivePolicy:
    """Tiny local backend for smoke tests; cycles valid/malformed responses."""

    backend_name = "fake"
    model_id = "fake-cycle-valid-and-malformed"

    def __init__(self) -> None:
        self._responses = [
            "<answer>Right</answer>",
            "<answer>Down</answer>",
            "Down",
            "<answer>Left</answer>",
            "<answer>Up</answer>",
            "<answer>Jump</answer>",
        ]
        self._index = 0

    def generate_action_text(self, frame: np.ndarray) -> str:
        del frame
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response


class QwenVLReactivePolicy:
    """Qwen2.5-VL Instruct backend for Colab GPU inference."""

    backend_name = "qwen2_5_vl"

    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 16,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch_dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16

        self._image_cls = Image
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )

    def generate_action_text(self, frame: np.ndarray) -> str:
        image = self._image_cls.fromarray(frame)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self._model.device)
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        decoded = self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


def make_policy(args: argparse.Namespace) -> ReactivePolicy:
    if args.backend == "fake":
        return FakeReactivePolicy()
    return QwenVLReactivePolicy(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )


def run_episode(seed: int, policy: ReactivePolicy, results_dir: Path, save_debug_frames: int) -> dict:
    desc = generate_reachable_random_map(seed=seed)
    env = make_frozenlake_env(desc, seed=seed)
    observation, info = env.reset(seed=seed)
    del info

    max_steps = int(getattr(env.spec, "max_episode_steps", 100) or 100)
    trace = []
    raw_responses = []
    total_reward = 0.0
    terminated = False
    truncated = False
    final_outcome = "step_cap"
    start_time = time.time()

    episode_dir = results_dir / f"seed_{seed:02d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "map.txt").write_text("\n".join(desc) + "\n", encoding="utf-8")

    for step in range(max_steps):
        frame = render_rgb_array(env)
        if step < save_debug_frames:
            np.save(episode_dir / f"frame_{step:03d}.npy", frame)

        raw_response = policy.generate_action_text(frame)
        parsed = parse_action_response(raw_response)
        raw_responses.append({"step": step, "raw_response": raw_response})

        prev_state = extract_gt_state(desc, observation)
        observation, reward, terminated, truncated, info = env.step(parsed.action_id)
        del info
        next_state = extract_gt_state(desc, observation)
        total_reward += float(reward)

        if terminated and reward > 0:
            final_outcome = "goal"
        elif terminated:
            final_outcome = "hole"
        elif truncated:
            final_outcome = "truncated"

        trace.append(
            {
                "step": step,
                "player_position": prev_state.player_position,
                "raw_response": raw_response,
                "format_compliant": parsed.compliant,
                "fallback_used": parsed.fallback_used,
                "parser_error": parsed.error,
                "action_name": parsed.action_name,
                "action_id": parsed.action_id,
                "gym_action_name": ACTION_NAMES[parsed.action_id],
                "reward": float(reward),
                "next_position": next_state.player_position,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            break

    env.close()
    steps = len(trace)
    compliant_steps = sum(1 for item in trace if item["format_compliant"])
    result = {
        "seed": seed,
        "backend": policy.backend_name,
        "model_id": policy.model_id,
        "map": desc,
        "max_episode_steps": max_steps,
        "steps": steps,
        "success": bool(final_outcome == "goal"),
        "final_outcome": final_outcome,
        "total_reward": total_reward,
        "format_compliant_steps": compliant_steps,
        "format_compliance_rate": compliant_steps / steps if steps else 0.0,
        "fallback_steps": sum(1 for item in trace if item["fallback_used"]),
        "runtime_seconds": time.time() - start_time,
        "trace": trace,
    }
    (episode_dir / "trace.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (episode_dir / "raw_responses.json").write_text(json.dumps(raw_responses, indent=2), encoding="utf-8")
    return result


def write_summary(results: list[dict], results_dir: Path) -> dict:
    episode_count = len(results)
    total_steps = sum(result["steps"] for result in results)
    success_count = sum(1 for result in results if result["success"])
    hole_count = sum(1 for result in results if result["final_outcome"] == "hole")
    truncated_count = sum(1 for result in results if result["final_outcome"] == "truncated")
    compliant_steps = sum(result["format_compliant_steps"] for result in results)
    summary = {
        "episodes": episode_count,
        "seeds": [result["seed"] for result in results],
        "backend": results[0]["backend"] if results else None,
        "model_id": results[0]["model_id"] if results else None,
        "success_count": success_count,
        "success_rate": success_count / episode_count if episode_count else 0.0,
        "hole_count": hole_count,
        "hole_rate": hole_count / episode_count if episode_count else 0.0,
        "truncated_count": truncated_count,
        "truncation_rate": truncated_count / episode_count if episode_count else 0.0,
        "mean_steps": total_steps / episode_count if episode_count else 0.0,
        "format_compliant_steps": compliant_steps,
        "total_steps": total_steps,
        "format_compliance_rate": compliant_steps / total_steps if total_steps else 0.0,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (results_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "success",
            "final_outcome",
            "steps",
            "total_reward",
            "format_compliance_rate",
            "fallback_steps",
            "runtime_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fieldnames})

    return summary


def run_parser_smoke() -> None:
    cases = [
        "<answer>Down</answer>",
        " <answer>Left</answer>\n",
        "<answer>Right</answer>",
        "<answer>Up</answer>",
        "Down",
        "<answer>Jump</answer>",
        "<answer>down</answer>",
        "The move is <answer>Down</answer>",
    ]
    print("Parser smoke")
    for text in cases:
        parsed = parse_action_response(text)
        print(
            f"input={text!r} compliant={parsed.compliant} "
            f"action={parsed.action_name}/{parsed.action_id} fallback={parsed.fallback_used}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["fake", "qwen"], default="fake")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--save-debug-frames", type=int, default=2)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--parser-smoke", action="store_true")
    args = parser.parse_args()

    if args.parser_smoke:
        run_parser_smoke()
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (args.results_dir / "user_prompt.txt").write_text(USER_PROMPT + "\n", encoding="utf-8")
    policy = make_policy(args)
    (args.results_dir / "run_config.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "model_id": policy.model_id,
                "requested_model_id": args.model_id,
                "seeds": args.seeds,
                "fallback_action": FALLBACK_ACTION_NAME,
                "fallback_action_id": FALLBACK_ACTION_ID,
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
                "max_new_tokens": args.max_new_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results = [run_episode(seed, policy, args.results_dir, args.save_debug_frames) for seed in args.seeds]
    summary = write_summary(results, args.results_dir)

    print("Stage A2 reactive VLM baseline")
    print(f"backend={policy.backend_name}")
    print(f"model_id={policy.model_id}")
    print(f"results_dir={args.results_dir}")
    print(f"seeds={summary['seeds']}")
    print(
        f"success_rate={summary['success_rate']:.3f} "
        f"hole_rate={summary['hole_rate']:.3f} "
        f"truncation_rate={summary['truncation_rate']:.3f} "
        f"mean_steps={summary['mean_steps']:.2f} "
        f"format_compliance_rate={summary['format_compliance_rate']:.3f}"
    )
    for result in results:
        print(
            f"seed={result['seed']} outcome={result['final_outcome']} "
            f"success={result['success']} steps={result['steps']} "
            f"compliance={result['format_compliance_rate']:.3f} "
            f"fallback_steps={result['fallback_steps']}"
        )
    print(f"summary_json={args.results_dir / 'summary.json'}")
    print(f"summary_csv={args.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
