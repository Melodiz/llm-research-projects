# activation patching experiments 1-5

import os, json, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import N_LAYERS, N_HEADS, N_DIGITS, SEED, ID_TO_CHAR
from patching import (
    compute_baselines, run_patching,
    patch_residual_stream, patch_component, patch_head,
    compute_logit_diff, compute_recovery,
    ANSWER_LOGIT_POS, ANSWER_SEQ_POS,
    _pattern_hook, _head_hook,
)
from data import encode_example, classify_example
from counterfactual import CARRY_TO_ANSWER_IDX

TOKEN_LABELS = [
    "a_H", "a_T", "a_O", "+", "b_H", "b_T", "b_O", "=",
    "s_Ov", "s_H", "s_T", "s_O",
]
LAYER_LABELS = ["Before L0", "Before L1", "After L1"]
ANSWER_LABELS = ["overflow", "hundreds", "tens", "ones"]
HEAD_LABELS = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]

N_INPUT_POS = 12
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _save_json(name, data):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


# exp 1 - residual stream patching heatmap
def experiment_1(model, pairs):
    print("\n--- Experiment 1: Residual stream patching heatmap ---")
    subtypes = ["type_a_ones", "type_a_tens", "type_a_hundreds"]
    n_layers_plus = N_LAYERS + 1  # 0, 1, 2 (before L0, before L1, after L1)

    all_results = {}
    for st in subtypes:
        plist = pairs[st]
        baselines = compute_baselines(model, plist)
        primary_ai = plist[0]["primary_answer_idx"]
        heatmap = np.zeros((n_layers_plus, N_INPUT_POS))

        for layer in range(n_layers_plus):
            for pos in range(N_INPUT_POS):
                res = run_patching(
                    model, plist, patch_residual_stream, baselines,
                    layer=layer, pos=pos,
                )
                heatmap[layer, pos] = res[primary_ai]["recovery_mean"]
            print(f"  {st} layer {layer} done")

        all_results[st] = heatmap.tolist()

    _save_json("exp1_residual_stream.json", all_results)
    print("  Saved exp1 results.")

    # find top positions
    summaries = []
    for st in subtypes:
        arr = np.array(all_results[st])
        top_idx = np.unravel_index(arr.argmax(), arr.shape)
        summaries.append(f"{st}: peak at layer={top_idx[0]} pos={top_idx[1]}({TOKEN_LABELS[top_idx[1]]}) "
                         f"recovery={arr[top_idx]:.3f}")
    return summaries


# exp 2 - component-level patching
def experiment_2(model, pairs):
    print("\n--- Experiment 2: Component-level patching ---")
    subtypes = ["type_a_ones", "type_a_tens", "type_a_hundreds"]
    components = [(0, "attn"), (0, "mlp"), (1, "attn"), (1, "mlp")]
    comp_labels = ["L0-Attn", "L0-MLP", "L1-Attn", "L1-MLP"]

    all_results = {}
    for st in subtypes:
        plist = pairs[st]
        baselines = compute_baselines(model, plist)
        st_results = {}

        for (layer, comp), label in zip(components, comp_labels):
            res = run_patching(
                model, plist, patch_component, baselines,
                component_type=comp, layer=layer,
            )
            st_results[label] = {
                ANSWER_LABELS[ai]: res[ai]["recovery_mean"] for ai in range(4)
            }
        all_results[st] = st_results
        print(f"  {st} done")

    _save_json("exp2_component_level.json", all_results)
    print("  Saved exp2 results.")

    ranked = []
    for st in subtypes:
        primary_ai = pairs[st][0]["primary_answer_idx"]
        primary_label = ANSWER_LABELS[primary_ai]
        scores = [(lab, all_results[st][lab][primary_label]) for lab in comp_labels]
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked.append(f"{st}: {', '.join(f'{l}={v:.3f}' for l, v in scores)}")
    return ranked

# exp 3 - head-level IE sweep (main result)
def experiment_3(model, pairs):
    # this is the main result from the paper
    print("\n--- Experiment 3: Head-level IE sweep ---")
    pair_types = ["type_a_ones", "type_a_tens", "type_a_hundreds", "type_b"]
    n_heads_total = N_LAYERS * N_HEADS

    all_results = {}
    for pt in pair_types:
        plist = pairs[pt]
        baselines = compute_baselines(model, plist)
        heatmap = np.zeros((n_heads_total, 4))

        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                hi = l * N_HEADS + h
                res = run_patching(
                    model, plist, patch_head, baselines,
                    layer=l, head=h,
                )
                for ai in range(4):
                    heatmap[hi, ai] = res[ai]["recovery_mean"]

        all_results[pt] = heatmap.tolist()
        print(f"  {pt} done")

    _save_json("exp3_head_level_ie.json", all_results)
    print("  Saved exp3 results.")

    summaries = []
    for pt in pair_types:
        data = np.array(all_results[pt])
        primary_ai = pairs[pt][0]["primary_answer_idx"]
        col = data[:, primary_ai]
        ranked = sorted(enumerate(col), key=lambda x: x[1], reverse=True)
        top3 = [(HEAD_LABELS[i], v) for i, v in ranked[:3]]
        summaries.append(f"{pt}: top heads at primary pos = "
                         + ", ".join(f"{h}={v:.3f}" for h, v in top3))
    return summaries


# exp 4 - attention pattern viz
def experiment_4(model):
    print("\n--- Experiment 4: Attention pattern visualization ---")

    examples = {
        "no_carry":      (123, 456),
        "carry_ones":    (137, 218),
        "carry_tens":    (152, 263),
        "carry_hundreds":(612, 503),
        "full_cascade":  (999,   1),
    }

    all_patterns = {}
    for name, (a, b) in examples.items():
        tokens = encode_example(a, b)
        inp = tokens[:-1].unsqueeze(0)
        token_strs = [ID_TO_CHAR.get(t.item(), "?") for t in tokens[:-1]]
        labels = []
        for i, ts in enumerate(token_strs):
            if i < 3:
                labels.append(f"{ts}")
            elif i == 3:
                labels.append("+")
            elif i < 7:
                labels.append(f"{ts}")
            elif i == 7:
                labels.append("=")
            else:
                labels.append(f"{ts}")

        _, cache = model.run_with_cache(inp)

        fig, axes = plt.subplots(N_LAYERS, N_HEADS, figsize=(4 * N_HEADS, 4 * N_LAYERS),
                                 constrained_layout=True)
        stored = {}
        for l in range(N_LAYERS):
            pattern = cache[_pattern_hook(l)][0]
            for h in range(N_HEADS):
                ax = axes[l, h] if N_LAYERS > 1 else axes[h]
                attn = pattern[h].detach().numpy()
                stored[f"L{l}H{h}"] = attn.tolist()
                im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_title(f"L{l}H{h}", fontsize=12)
                if h == N_HEADS - 1:
                    fig.colorbar(im, ax=ax, shrink=0.7)

        fig.savefig(os.path.join(RESULTS_DIR, f"exp4_attention_patterns_{name}.png"),
                    bbox_inches="tight")
        plt.close(fig)
        all_patterns[name] = {"a": a, "b": b, "patterns": stored}

    _save_json("exp4_attention_patterns.json", {
        k: {"a": v["a"], "b": v["b"]} for k, v in all_patterns.items()
    })
    print("  Saved exp4 results.")

    staircase_heads = []
    for name, info in all_patterns.items():
        for head_name, attn in info["patterns"].items():
            attn = np.array(attn)
            # s_O->a_O+b_O, s_T->a_T+b_T, s_H->a_H+b_H
            operand_map = {11: [2, 6], 10: [1, 5], 9: [0, 4]}
            is_staircase = True
            for qpos, kpositions in operand_map.items():
                if qpos < attn.shape[0]:
                    attn_to_operands = sum(attn[qpos, kp] for kp in kpositions)
                    if attn_to_operands < 0.3:
                        is_staircase = False
                        break
            if is_staircase and head_name not in staircase_heads:
                staircase_heads.append(head_name)

    return [f"Staircase-like heads: {staircase_heads if staircase_heads else 'none detected'}"]

# exp 5 - direct logit attribution
def experiment_5(model, pairs):
    print("\n--- Experiment 5: Direct logit attribution ---")
    random.seed(SEED)

    carry_pairs = pairs["type_a_ones"][:200]
    no_carry_examples = []
    for p in pairs["type_a_ones"][:200]:
        no_carry_examples.append({
            "clean_tokens": p["corrupted_tokens"],
            "corrupted_tokens": p["clean_tokens"],
        })

    W_U = model.W_U.detach()

    results = {"carry": {}, "no_carry": {}}

    for label, examples in [("carry", carry_pairs), ("no_carry", no_carry_examples)]:
        head_votes = {}

        batch_size = 64
        for s in range(0, len(examples), batch_size):
            batch = examples[s:s + batch_size]
            tokens = torch.tensor([ex["clean_tokens"] for ex in batch], dtype=torch.long)
            inp = tokens[:, :-1]
            _, cache = model.run_with_cache(inp)

            for l in range(N_LAYERS):
                hook = f"blocks.{l}.attn.hook_result"
                head_out = cache[hook]
                for h in range(N_HEADS):
                    hname = f"L{l}H{h}"
                    for ai in range(4):
                        lp = ANSWER_LOGIT_POS[ai]
                        h_vec = head_out[:, lp, h, :]
                        logit_contrib = h_vec @ W_U
                        # only look at digit logits 0-9
                        digit_logits = logit_contrib[:, :10]
                        top_digit = digit_logits.argmax(dim=1)

                        key = (hname, ai)
                        if key not in head_votes:
                            head_votes[key] = []
                        head_votes[key].extend(top_digit.tolist())

        mode_table = {}
        for (hname, ai), votes in head_votes.items():
            from statistics import mode as stat_mode
            try:
                m = stat_mode(votes)
            except Exception:
                m = max(set(votes), key=votes.count)
            freq = votes.count(m) / len(votes)
            if hname not in mode_table:
                mode_table[hname] = {}
            mode_table[hname][ANSWER_LABELS[ai]] = {"mode_digit": m, "frequency": freq}

        results[label] = mode_table

    _save_json("exp5_logit_attribution.json", results)
    print("  Saved exp5 results.")

    ba_heads = []
    carry_heads = []
    for hname in HEAD_LABELS:
        differs = False
        for aname in ANSWER_LABELS:
            c_entry = results["carry"].get(hname, {}).get(aname, {})
            nc_entry = results["no_carry"].get(hname, {}).get(aname, {})
            if c_entry.get("mode_digit") != nc_entry.get("mode_digit"):
                differs = True
                break
        if differs:
            carry_heads.append(hname)
        else:
            ba_heads.append(hname)

    return [f"BA heads (same vote): {ba_heads}", f"Carry-sensitive heads: {carry_heads}"]
