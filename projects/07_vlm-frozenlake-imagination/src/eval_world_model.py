"""evaluate SFT FrozenLake world-model adapters"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_sft import Condition, build_messages, load_jsonl, load_model_and_processor, user_text

DEFAULT_DATA_DIR = Path("results/A3_dataset_collection")
DEFAULT_ADAPTER_ROOT = Path("results/A4_sft_training")
DEFAULT_OUTPUT_ROOT = Path("results/A5_world_model_eval")
OUTCOMES = ("safe", "hole", "goal", "wall")
TARGET_RE = re.compile(
    r"^<prediction>Position: \((\d+), (\d+)\)\. Outcome: (safe|hole|goal|wall)</prediction>$"
)


def parse_prediction(text: str) -> dict[str, Any]:
    stripped = text.strip()
    match = TARGET_RE.fullmatch(stripped)
    if not match:
        return {
            "parse_compliant": False,
            "position": None,
            "outcome": None,
            "error": "expected exactly <prediction>Position: (r, c). Outcome: {safe|hole|goal|wall}</prediction>",
        }
    return {
        "parse_compliant": True,
        "position": [int(match.group(1)), int(match.group(2))],
        "outcome": match.group(3),
        "error": None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_confusion_csv(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*OUTCOMES, "format_error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_outcome", *columns])
        for target in OUTCOMES:
            writer.writerow([target, *[matrix[target].get(pred, 0) for pred in columns]])


def load_adapter_model(args: argparse.Namespace):
    from peft import PeftModel

    model, processor = load_model_and_processor(args)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model, processor


def generate_one(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    row: dict[str, Any],
    condition: Condition,
    max_new_tokens: int,
) -> str:
    messages = build_messages(row, condition, include_answer=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    decoded = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def evaluate_rows(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    rows: list[dict[str, Any]],
    condition: Condition,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        generated_text = generate_one(
            model=model,
            processor=processor,
            process_vision_info=process_vision_info,
            row=row,
            condition=condition,
            max_new_tokens=max_new_tokens,
        )
        parsed = parse_prediction(generated_text)
        target_parsed = parse_prediction(row["target_text"])
        if not target_parsed["parse_compliant"]:
            raise AssertionError(f"A3 target text failed parser: {row['transition_id']}")

        predicted_position = parsed["position"]
        predicted_outcome = parsed["outcome"]
        target_position = target_parsed["position"]
        target_outcome = target_parsed["outcome"]
        position_correct = bool(parsed["parse_compliant"] and predicted_position == target_position)
        outcome_correct = bool(parsed["parse_compliant"] and predicted_outcome == target_outcome)
        both_correct = bool(position_correct and outcome_correct)

        predictions.append(
            {
                "row_index": index,
                "transition_id": row["transition_id"],
                "split": row["split"],
                "condition": condition,
                "policy_source": row["policy_source"],
                "image_path": row["image_path"],
                "action_id": row["action_id"],
                "action_name": row["action_name"],
                "prompt_user_text": user_text(row, condition),
                "target_text": row["target_text"],
                "generated_text": generated_text,
                "parse_compliant": parsed["parse_compliant"],
                "parse_error": parsed["error"],
                "target_position": target_position,
                "predicted_position": predicted_position,
                "target_outcome": target_outcome,
                "predicted_outcome": predicted_outcome,
                "exact_target_match": generated_text.strip() == row["target_text"],
                "position_correct": position_correct,
                "outcome_correct": outcome_correct,
                "both_correct": both_correct,
                "gt_state_text_present_in_prompt": row["gt_state_text"] in user_text(row, condition),
            }
        )
    return predictions


def metrics_for_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(predictions)
    compliant = sum(row["parse_compliant"] for row in predictions)
    exact = sum(row["exact_target_match"] for row in predictions)
    position = sum(row["position_correct"] for row in predictions)
    outcome = sum(row["outcome_correct"] for row in predictions)
    both = sum(row["both_correct"] for row in predictions)
    format_errors = count - compliant

    wrong_position = sum(row["parse_compliant"] and not row["position_correct"] for row in predictions)
    wrong_outcome = sum(row["parse_compliant"] and not row["outcome_correct"] for row in predictions)
    wrong_both = sum(
        row["parse_compliant"] and not row["position_correct"] and not row["outcome_correct"]
        for row in predictions
    )

    by_policy = {}
    for policy in sorted({row["policy_source"] for row in predictions}):
        subset = [row for row in predictions if row["policy_source"] == policy]
        by_policy[policy] = simple_rates(subset)

    by_target_outcome = {}
    for outcome_name in OUTCOMES:
        subset = [row for row in predictions if row["target_outcome"] == outcome_name]
        by_target_outcome[outcome_name] = simple_rates(subset)

    return {
        "count": count,
        "format_compliant": compliant,
        "format_error_count": format_errors,
        "format_compliance_rate": compliant / count if count else 0.0,
        "exact_target_match_count": exact,
        "exact_target_match_rate": exact / count if count else 0.0,
        "position_correct_count": position,
        "position_accuracy": position / count if count else 0.0,
        "outcome_correct_count": outcome,
        "outcome_accuracy": outcome / count if count else 0.0,
        "both_correct_count": both,
        "both_correct_accuracy": both / count if count else 0.0,
        "wrong_position_count": wrong_position,
        "wrong_outcome_count": wrong_outcome,
        "wrong_both_count": wrong_both,
        "by_policy_source": by_policy,
        "by_target_outcome": by_target_outcome,
    }


def simple_rates(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(predictions)
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "format_compliance_rate": sum(row["parse_compliant"] for row in predictions) / count,
        "exact_target_match_rate": sum(row["exact_target_match"] for row in predictions) / count,
        "position_accuracy": sum(row["position_correct"] for row in predictions) / count,
        "outcome_accuracy": sum(row["outcome_correct"] for row in predictions) / count,
        "both_correct_accuracy": sum(row["both_correct"] for row in predictions) / count,
    }


def confusion_matrix(predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        target: {pred: 0 for pred in [*OUTCOMES, "format_error"]} for target in OUTCOMES
    }
    for row in predictions:
        target = row["target_outcome"]
        pred = row["predicted_outcome"] if row["parse_compliant"] else "format_error"
        matrix[target][pred] += 1
    return matrix


def representative_errors(predictions: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    errors = [row for row in predictions if not row["both_correct"] or not row["parse_compliant"]]
    return errors[:limit]


def assert_prompt_condition(predictions: list[dict[str, Any]], condition: Condition) -> None:
    if condition == "image_text":
        bad = [row["transition_id"] for row in predictions if not row["gt_state_text_present_in_prompt"]]
    else:
        bad = [row["transition_id"] for row in predictions if row["gt_state_text_present_in_prompt"]]
    if bad:
        raise AssertionError(f"prompt condition check failed for {condition}: {bad[:5]}")


def run_parser_smoke() -> None:
    cases = [
        "<prediction>Position: (0, 1). Outcome: safe</prediction>",
        " <prediction>Position: (7, 7). Outcome: goal</prediction>\n",
        "<prediction>Position: (3, 4). Outcome: wall</prediction>",
        "<prediction>Position: (1, 2). Outcome: lava</prediction>",
        "Position: (0, 1). Outcome: safe",
    ]
    print("Stage A5 parser smoke")
    for text in cases:
        parsed = parse_prediction(text)
        print(
            f"input={text!r} compliant={parsed['parse_compliant']} "
            f"position={parsed['position']} outcome={parsed['outcome']}"
        )


def run_prompt_smoke(args: argparse.Namespace) -> None:
    for condition in ("image_text", "image_only"):
        rows = load_jsonl(args.data_dir / "train.jsonl", limit=1)
        prompt = user_text(rows[0], condition)  # type: ignore[arg-type]
        contains_gt = rows[0]["gt_state_text"] in prompt
        print(f"condition={condition} contains_gt_state_text={contains_gt}")
        print(prompt)


def run_eval(args: argparse.Namespace) -> None:
    import torch
    from qwen_vl_utils import process_vision_info

    args.adapter_dir = args.adapter_root / args.condition / "adapter"
    output_dir = args.output_root / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.data_dir / "train.jsonl", limit=args.train_limit)
    val_rows = load_jsonl(args.data_dir / "val.jsonl", limit=args.val_limit)
    model, processor = load_adapter_model(args)

    start = time.time()
    train_predictions = evaluate_rows(
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        rows=train_rows,
        condition=args.condition,
        max_new_tokens=args.max_new_tokens,
    )
    val_predictions = evaluate_rows(
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        rows=val_rows,
        condition=args.condition,
        max_new_tokens=args.max_new_tokens,
    )
    runtime_seconds = time.time() - start

    assert_prompt_condition(train_predictions[: min(20, len(train_predictions))], args.condition)
    assert_prompt_condition(val_predictions[: min(20, len(val_predictions))], args.condition)

    train_metrics = metrics_for_predictions(train_predictions)
    val_metrics = metrics_for_predictions(val_predictions)
    train_confusion = confusion_matrix(train_predictions)
    val_confusion = confusion_matrix(val_predictions)
    calibration = (
        "converged_enough"
        if train_metrics["both_correct_accuracy"] >= args.train_calibration_threshold
        else "not_converged_do_not_make_strong_val_claims"
    )

    error_examples = [
        {"split": "train", **row} for row in representative_errors(train_predictions, args.error_examples)
    ] + [{"split": "val", **row} for row in representative_errors(val_predictions, args.error_examples)]

    write_jsonl(output_dir / "predictions_train.jsonl", train_predictions)
    write_jsonl(output_dir / "predictions_val.jsonl", val_predictions)
    write_jsonl(output_dir / "error_examples.jsonl", error_examples)
    write_json(output_dir / "confusion_matrix_train.json", train_confusion)
    write_json(output_dir / "confusion_matrix_val.json", val_confusion)
    write_confusion_csv(output_dir / "confusion_matrix_train.csv", train_confusion)
    write_confusion_csv(output_dir / "confusion_matrix_val.csv", val_confusion)

    config = {
        "condition": args.condition,
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir),
        "data_dir": str(args.data_dir),
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "max_new_tokens": args.max_new_tokens,
        "bf16": args.bf16,
        "load_in_4bit": args.load_in_4bit,
        "device_map": args.device_map,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    write_json(output_dir / "run_config.json", config)
    write_json(
        output_dir / "metrics.json",
        {
            "condition": args.condition,
            "runtime_seconds": runtime_seconds,
            "train": train_metrics,
            "val": val_metrics,
            "train_calibration_threshold": args.train_calibration_threshold,
            "train_calibration_verdict": calibration,
            "prediction_counts": {
                "train": len(train_predictions),
                "val": len(val_predictions),
            },
        },
    )

    expected_train = args.train_limit or 2400
    expected_val = args.val_limit or 600
    if len(train_predictions) != expected_train:
        raise AssertionError(f"train prediction count mismatch: {len(train_predictions)} != {expected_train}")
    if len(val_predictions) != expected_val:
        raise AssertionError(f"val prediction count mismatch: {len(val_predictions)} != {expected_val}")

    print("Stage A5 world-model eval")
    print(f"condition={args.condition}")
    print(f"adapter_dir={args.adapter_dir}")
    print(f"output_dir={output_dir}")
    print(f"gpu_name={config['gpu_name']}")
    print(f"train_predictions={len(train_predictions)} val_predictions={len(val_predictions)}")
    print(f"train_metrics={train_metrics}")
    print(f"val_metrics={val_metrics}")
    print(f"train_calibration_verdict={calibration}")
    print(f"runtime_seconds={runtime_seconds:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parser-smoke", "prompt-smoke", "eval"], default="parser-smoke")
    parser.add_argument("--condition", choices=["image_text", "image_only"], default="image_text")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--train-calibration-threshold", type=float, default=0.9)
    parser.add_argument("--error-examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "parser-smoke":
        run_parser_smoke()
    elif args.mode == "prompt-smoke":
        run_prompt_smoke(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        raise ValueError(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()
