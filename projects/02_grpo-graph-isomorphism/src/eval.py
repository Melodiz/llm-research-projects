import os
import re
import json
import csv
import argparse
from collections import defaultdict


import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_isomorphism_env import GraphIsomorphismVerifier, Data

_FMT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>.*?</answer>",
    re.DOTALL | re.IGNORECASE,
)

def compute_metrics(
    completions,
    ground_truths,
    metadata_list,
    **kwargs
):
    verifier = GraphIsomorphismVerifier()
    total = len(completions)
    if total == 0:
        return {}

    iso_count = 0
    iso_correct = 0
    isa_comp_len = 0

    noniso_count = 0
    noniso_correct = 0
    noniso_comp_len = 0

    not_iso_predictions = 0
    format_valid = 0

    err_format_fail = 0
    err_wrong_mapping = 0
    err_false_not_iso = 0
    err_false_mapping_claim = 0

    for comp, gt, meta in zip(completions, ground_truths, metadata_list):
        is_iso = meta.get("is_isomorphic", True)
        tok_len = len(comp.split())
        data = Data(question="", answer=gt, metadata=meta)
        correct = verifier.verify(data, comp)
        has_fmt = bool(_FMT_RE.search(comp))

        if has_fmt:
            format_valid += 1

        ans = verifier.extract_answer(comp)
        claims_not_iso = verifier._is_not_isomorphic_declaration(ans)
        if claims_not_iso:
            not_iso_predictions += 1

        if is_iso:
            iso_count += 1
            isa_comp_len += tok_len
            if correct:
                iso_correct += 1
            else:
                if not ans:
                    err_format_fail += 1
                elif claims_not_iso:
                    err_false_not_iso += 1
                else:
                    err_wrong_mapping += 1
        else:
            noniso_count += 1
            noniso_comp_len += tok_len
            if correct:
                noniso_correct += 1
            else:
                if not ans:
                    err_format_fail += 1
                else:
                    err_false_mapping_claim += 1

    agg_correct = iso_correct + noniso_correct

    return {
        "total": total,
        "iso_count": iso_count,
        "noniso_count": noniso_count,
        "aggregate_accuracy": agg_correct / total if total > 0 else 0.0,
        "iso_accuracy": iso_correct / iso_count if iso_count > 0 else 0.0,
        "non_iso_accuracy": noniso_correct / noniso_count if noniso_count > 0 else 0.0,
        "class_prediction_ratio": not_iso_predictions / total if total > 0 else 0.0,
        "format_compliance": format_valid / total if total > 0 else 0.0,
        "mean_response_length_iso": isa_comp_len / iso_count if iso_count > 0 else 0.0,
        "mean_response_length_non_iso": noniso_comp_len / noniso_count if noniso_count > 0 else 0.0,
        "err_format_fail": err_format_fail,
        "err_wrong_mapping": err_wrong_mapping,
        "err_false_not_iso": err_false_not_iso,
        "err_false_mapping_claim": err_false_mapping_claim,
        "degenerate_floor": noniso_count / total if total > 0 else 0.0,
    }


def run_evaluation(model_path, test_dir, out_dir, difficulties):
    import torch
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    os.makedirs(out_dir, exist_ok=True)

    instances = []
    for d in difficulties:
        mf = os.path.join(test_dir, f"mixed_diff{d}.jsonl")
        if os.path.exists(mf):
            with open(mf, "r") as f:
                for line in f:
                    instances.append(json.loads(line))
    
    for d in difficulties:
        if_ = os.path.join(test_dir, f"iso_only_diff{d}.jsonl")
        if os.path.exists(if_):
            with open(if_, "r") as f:
                for line in f:
                    instances.append(json.loads(line))

    if not instances:
        print("no test instances found")
        return

    print(f"{len(instances)} instances")

    # Always load base model first
    print("Loading base model...")
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

    SYSTEM_PROMPT = (
        "Respond in the following format:\n"
        "<think>\n...\n</think>\n"
        "<answer>\n...\n</answer>"
    )
    
    prompts = []
    for inst in instances:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inst["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    print("generating...")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )
    
    outputs = model.fast_generate(
        prompts,
        sampling_params=sampling_params,

    )
    
    results_by_diff = defaultdict(list)
    raw_results = []
    
    for inst, out in zip(instances, outputs):
        comp = out.outputs[0].text
        # base model outputs bare text without tags — wrap so verifier can parse;
        # _FMT_RE still fails (no <think>) so format_compliance stays 0%
        if "<answer>" not in comp.lower():
            comp = f"<answer>{comp}</answer>"
        res = {
            "difficulty": inst["difficulty"],
            "is_isomorphic": inst["metadata"]["is_isomorphic"],
            "question": inst["question"],
            "ground_truth": inst["answer"],
            "completion": comp,
            "metadata": inst["metadata"]
        }
        results_by_diff[inst["difficulty"]].append(res)
        raw_results.append(res)
        
    metrics_path = os.path.join(out_dir, "metrics_per_difficulty.csv")
    tax_path = os.path.join(out_dir, "error_taxonomy.csv")
    deg_path = os.path.join(out_dir, "degenerate_floor.csv")
    raw_path = os.path.join(out_dir, "raw_completions.jsonl")
    samp_path = os.path.join(out_dir, "completions_sample.jsonl")

    with open(metrics_path, "w", newline="") as fm, \
         open(tax_path, "w", newline="") as ft, \
         open(deg_path, "w", newline="") as fd:
             
        metric_fields = ["difficulty", "total", "iso_count", "noniso_count",
                         "aggregate_accuracy", "iso_accuracy", "non_iso_accuracy",
                         "class_prediction_ratio", "format_compliance",
                         "mean_response_length_iso", "mean_response_length_non_iso"]
        mw = csv.DictWriter(fm, fieldnames=metric_fields)
        mw.writeheader()
        
        tax_fields = ["difficulty", "err_format_fail", "err_wrong_mapping", 
                      "err_false_not_iso", "err_false_mapping_claim"]
        tw = csv.DictWriter(ft, fieldnames=tax_fields)
        tw.writeheader()
        
        dw = csv.writer(fd)
        dw.writerow(["difficulty", "degenerate_floor"])

        for d in sorted(results_by_diff.keys()):
            r = results_by_diff[d]
            comps = [x["completion"] for x in r]
            gts = [x["ground_truth"] for x in r]
            metas = [x["metadata"] for x in r]
            m = compute_metrics(comps, gts, metas)
            
            m["difficulty"] = d; mw.writerow({k: m.get(k, "") for k in metric_fields})
            tw.writerow({
                "difficulty": d,
                "err_format_fail": m["err_format_fail"],
                "err_wrong_mapping": m["err_wrong_mapping"],
                "err_false_not_iso": m["err_false_not_iso"],
                "err_false_mapping_claim": m["err_false_mapping_claim"]
            })
            dw.writerow([d, m["degenerate_floor"]])

    with open(raw_path, "w") as f:
        for r in raw_results:
            f.write(json.dumps(r) + "\n")

    with open(samp_path, "w") as f:
        for d in sorted(results_by_diff.keys()):
            subset = results_by_diff[d][:10]
            for r in subset:
                f.write(json.dumps(r) + "\n")

    print(f"done → {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/test_sets")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--difficulties", type=str, default="1,2,3,4,5,6,7,8,9,10")
    args = parser.parse_args()

    diffs = [int(x.strip()) for x in args.difficulties.split(",")]
    run_evaluation(args.model, args.test_dir, args.output_dir, diffs)

if __name__ == "__main__":
    main()
