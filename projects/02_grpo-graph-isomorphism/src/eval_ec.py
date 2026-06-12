"""Evaluation script for edge counting positive control."""

import os
import re
import csv
import json
import argparse

from edge_counting_env import EdgeCountingEnv, EdgeCountingVerifier, SYSTEM_PROMPT
from graph_isomorphism_env import Data

_FMT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>.*?</answer>",
    re.DOTALL | re.IGNORECASE,
)


def generate_test_set(difficulties=(1, 2, 3), num_per_diff=50, seed=9999):
    """Generate edge counting test problems on-the-fly."""
    env = EdgeCountingEnv()
    instances = []
    for d in difficulties:
        data_list = env.generate(
            num_of_questions=num_per_diff,
            difficulty=d,
            seed=seed + d,
        )
        for item in data_list:
            instances.append({
                "question": item.question,
                "answer": item.answer,
                "difficulty": d,
                "metadata": item.metadata,
            })
    return instances


def run_evaluation(model_path, out_dir, difficulties=(1, 2, 3),
                   num_per_diff=50):
    import torch
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    os.makedirs(out_dir, exist_ok=True)

    instances = generate_test_set(difficulties, num_per_diff)
    print(f"{len(instances)} test instances generated")

    # Always load base model first
    print(f"Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.6,
    )

    # If adapter path provided, load PEFT adapter on top
    if model_path != "base":
        from peft import PeftModel
        print(f"Loading adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path)

    FastLanguageModel.for_inference(model)
    print("Model ready for inference")

    prompts = []
    for inst in instances:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inst["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)

    print("Generating...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    outputs = model.fast_generate(prompts, sampling_params=sampling_params)

    verifier = EdgeCountingVerifier()


    from collections import defaultdict
    results_by_diff = defaultdict(list)

    for inst, out in zip(instances, outputs):
        comp = out.outputs[0].text
        # If model doesn't produce answer tags, wrap for verifier
        if "<answer>" not in comp.lower():
            comp = f"<answer>{comp}</answer>"

        data = Data(
            question="", answer=inst["answer"],
            metadata=inst["metadata"],
        )
        correct = verifier.verify(data, comp)
        has_fmt = bool(_FMT_RE.search(comp))
        resp_len = len(comp.split())

        results_by_diff[inst["difficulty"]].append({
            "correct": correct,
            "has_fmt": has_fmt,
            "resp_len": resp_len,
            "completion": comp,
            "expected": inst["answer"],
        })

    # Write CSV
    csv_path = os.path.join(out_dir, "ec_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["difficulty", "accuracy", "format_compliance",
                         "mean_response_length"])
        for d in sorted(results_by_diff.keys()):
            r = results_by_diff[d]
            n = len(r)
            acc = sum(1 for x in r if x["correct"]) / n
            fmt = sum(1 for x in r if x["has_fmt"]) / n
            avg_len = sum(x["resp_len"] for x in r) / n
            writer.writerow([d, f"{acc:.4f}", f"{fmt:.4f}", f"{avg_len:.1f}"])
            print(f"  Diff {d}: accuracy={acc:.1%}, format={fmt:.1%}, "
                  f"avg_len={avg_len:.0f}")

    # Write raw completions
    raw_path = os.path.join(out_dir, "ec_raw_completions.jsonl")
    with open(raw_path, "w") as f:
        for d in sorted(results_by_diff.keys()):
            for r in results_by_diff[d]:
                f.write(json.dumps({
                    "difficulty": d,
                    "expected": r["expected"],
                    "correct": r["correct"],
                    "completion": r["completion"],
                }) + "\n")

    print(f"Done → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/eval_ec")
    parser.add_argument("--num-per-diff", type=int, default=50)
    args = parser.parse_args()

    run_evaluation(args.model, args.output_dir, num_per_diff=args.num_per_diff)


if __name__ == "__main__":
    main()
