# circuit hypothesis + validation helpers
import json, os, random
import numpy as np
import torch
from itertools import combinations

from config import (
    N_LAYERS, N_HEADS, D_MODEL, ANSWER_START, SEQ_LEN, N_DIGITS, SEED,
)
from data import encode_example, load_test_sets
from counterfactual import ANSWER_LOGIT_POS, ANSWER_SEQ_POS
from patching import _head_hook, _pattern_hook

ALL_HEADS = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
ALL_MLPS = [(l, "mlp") for l in range(N_LAYERS)]
ALL_COMPONENTS = [(l, h) for l, h in ALL_HEADS] + ALL_MLPS
HEAD_LABELS = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
MLP_LABELS = [f"L{l}-MLP" for l in range(N_LAYERS)]
ALL_LABELS = HEAD_LABELS + MLP_LABELS
ANSWER_LABELS_LIST = ["overflow", "hundreds", "tens", "ones"]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")

N_INPUT_POS = SEQ_LEN - 1  # 12


def define_circuit_from_results(exp3_path=None):
    if exp3_path is None:
        exp3_path = os.path.join(RESULTS_DIR, "exp3_head_level_ie.json")

    with open(exp3_path) as f:
        exp3 = json.load(f)

    primary_idx = {
        "type_a_ones": 2,
        "type_a_tens": 1,
        "type_a_hundreds": 0,
        "type_b": 1,
    }

    head_ies = {}
    for hi, (l, h) in enumerate(ALL_HEADS):
        label = f"L{l}H{h}"
        head_ies[label] = {}
        for pt, pi in primary_idx.items():
            head_ies[label][pt] = exp3[pt][hi][pi]

    ranking = []
    for label in HEAD_LABELS:
        ies = head_ies[label]
        ie_values = list(ies.values())
        mean_ie = np.mean(ie_values)
        max_ie = np.max(ie_values)
        ranking.append({
            "head": label,
            "mean_ie": float(mean_ie),
            "max_ie": float(max_ie),
            "per_type": {k: float(v) for k, v in ies.items()},
        })

    ranking.sort(key=lambda x: x["max_ie"], reverse=True)

    print("\n" + "=" * 60)
    print("HEAD RANKING BY IE AT PRIMARY ANSWER POSITION")
    print("=" * 60)
    print(f"{'Head':<8} {'Mean IE':>10} {'Max IE':>10}  {'A-ones':>10} {'A-tens':>10} {'A-hund':>10} {'B-casc':>10}")
    print("-" * 60)
    for r in ranking:
        pt = r["per_type"]
        print(f"{r['head']:<8} {r['mean_ie']:>10.4f} {r['max_ie']:>10.4f}  "
              f"{pt['type_a_ones']:>10.4f} {pt['type_a_tens']:>10.4f} "
              f"{pt['type_a_hundreds']:>10.4f} {pt['type_b']:>10.4f}")

    threshold = 0.3
    circuit_heads = [r["head"] for r in ranking if r["max_ie"] > threshold]

    print(f"\nThreshold: max IE > {threshold}")
    print(f"Circuit heads: {circuit_heads}")

    def parse_head(label):
        return (int(label[1]), int(label[3]))

    circuit_head_tuples = [parse_head(h) for h in circuit_heads]

    primary = []
    secondary = []
    for label in circuit_heads:
        l, h = parse_head(label)
        if l == 0:
            primary.append((l, h))
        else:
            secondary.append((l, h))

    circuit_mlps = [(1, "mlp")]

    CARRY_CIRCUIT = {
        "primary": primary,
        "secondary": secondary,
        "mlp": circuit_mlps,
    }

    circuit_flat = list(primary) + list(secondary) + list(circuit_mlps)

    non_circuit = []
    for l, h in ALL_HEADS:
        if (l, h) not in circuit_head_tuples:
            non_circuit.append((l, h))
    for l, _ in ALL_MLPS:
        if (l, "mlp") not in circuit_mlps:
            non_circuit.append((l, "mlp"))

    print(f"\nCircuit definition:")
    print(f"  Primary (L0 heads):   {[(f'L{l}H{h}') for l, h in primary]}")
    print(f"  Secondary (L1 heads): {[(f'L{l}H{h}') for l, h in secondary]}")
    print(f"  MLP:                  {[(f'L{l}-MLP') for l, _ in circuit_mlps]}")
    print(f"  Total: {len(circuit_flat)} of {len(ALL_COMPONENTS)} components "
          f"({100*len(circuit_flat)/len(ALL_COMPONENTS):.0f}%)")
    print(f"\nNon-circuit: {_comp_labels(non_circuit)}")

    return CARRY_CIRCUIT, circuit_flat, non_circuit, ranking


def _comp_label(comp):
    l, c = comp
    if isinstance(c, int):
        return f"L{l}H{c}"
    return f"L{l}-MLP"

def _comp_labels(comps):
    return [_comp_label(c) for c in comps]


def compute_mean_activations(model, n_examples=2000, batch_size=64):
    # computes per-position mean activations for ablation baseline
    random.seed(SEED)
    from data import generate_balanced_dataset

    examples = generate_balanced_dataset(n_examples)
    tokens_list = [encode_example(a, b) for a, b, _ in examples]
    all_tokens = torch.stack(tokens_list)
    all_input = all_tokens[:, :-1]

    hook_names = []
    for l in range(N_LAYERS):
        hook_names.append(f"blocks.{l}.attn.hook_result")
        hook_names.append(f"blocks.{l}.hook_mlp_out")

    sums = {h: None for h in hook_names}
    count = 0

    with torch.no_grad():
        for start in range(0, len(all_input), batch_size):
            batch = all_input[start:start + batch_size]
            _, cache = model.run_with_cache(batch)
            bs = batch.shape[0]
            count += bs

            for h in hook_names:
                act = cache[h]
                if sums[h] is None:
                    sums[h] = act.sum(dim=0)
                else:
                    sums[h] += act.sum(dim=0)

    means = {h: sums[h] / count for h in hook_names}
    return means


def _ablate_and_evaluate(model, test_data, mean_acts, components_to_ablate):
    # run model with mean-ablated components, return per-cat accuracy
    hooks = []
    for l, c in components_to_ablate:
        if isinstance(c, int):
            hook_name = f"blocks.{l}.attn.hook_result"
            head_idx = c
            mean_val = mean_acts[hook_name]

            def make_head_hook(mean_v, hi):
                def hook_fn(activation, **kwargs):
                    activation[:, :, hi, :] = mean_v[:, hi, :].unsqueeze(0)
                    return activation
                return hook_fn

            hooks.append((hook_name, make_head_hook(mean_val, head_idx)))
        else:
            hook_name = f"blocks.{l}.hook_mlp_out"
            mean_val = mean_acts[hook_name]

            def make_mlp_hook(mean_v):
                def hook_fn(activation, **kwargs):
                    activation[:] = mean_v.unsqueeze(0)
                    return activation
                return hook_fn

            hooks.append((hook_name, make_mlp_hook(mean_val)))

    model.eval()
    results = {}
    digit_names = ["thousands(overflow)", "hundreds", "tens", "ones"]

    for cat_name, examples in test_data.items():
        n = len(examples)
        correct_full = 0
        correct_per_digit = [0] * (N_DIGITS + 1)
        total = 0

        for start in range(0, n, 256):
            batch_examples = examples[start:start + 256]
            tokens_list = [encode_example(a, b) for a, b, _ in batch_examples]
            tokens = torch.stack(tokens_list)
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            bs = input_tokens.shape[0]

            with torch.no_grad():
                logits = model.run_with_hooks(input_tokens, fwd_hooks=hooks)

            preds = logits.argmax(dim=-1)
            answer_preds = preds[:, ANSWER_START - 1:][:, :N_DIGITS + 1]
            answer_targets = target_tokens[:, ANSWER_START - 1:][:, :N_DIGITS + 1]

            full_correct = (answer_preds == answer_targets).all(dim=1)
            correct_full += full_correct.sum().item()
            for d in range(N_DIGITS + 1):
                correct_per_digit[d] += (answer_preds[:, d] == answer_targets[:, d]).sum().item()
            total += bs

        results[cat_name] = {
            "accuracy": correct_full / total,
            "per_digit": {
                digit_names[d]: correct_per_digit[d] / total
                for d in range(N_DIGITS + 1)
            },
            "total": total,
        }

    return results

def test_necessity(model, test_data, circuit_flat, non_circuit, mean_acts):
    print("\n" + "=" * 60)
    print("NECESSITY TESTING")
    print("=" * 60)

    print("\nComputing baseline accuracy...")
    baseline = _ablate_and_evaluate(model, test_data, mean_acts, [])

    results = {"baseline": baseline, "individual": {}, "group": {}}

    all_comps = circuit_flat + non_circuit
    for comp in all_comps:
        label = _comp_label(comp)
        print(f"  Ablating {label}...")
        res = _ablate_and_evaluate(model, test_data, mean_acts, [comp])
        results["individual"][label] = res

    print("  Ablating ALL circuit components...")
    results["group"]["all_circuit"] = _ablate_and_evaluate(
        model, test_data, mean_acts, circuit_flat)

    print("  Ablating ALL non-circuit components...")
    results["group"]["all_non_circuit"] = _ablate_and_evaluate(
        model, test_data, mean_acts, non_circuit)

    categories = ["ba", "mc1", "us9", "s3"]
    cat_display = {"ba": "BA", "mc1": "MC1", "us9": "US9", "s3": "S3"}

    print(f"\n{'Component Ablated':<20}", end="")
    for cat in categories:
        print(f" {cat_display[cat]+' Acc':>10}", end="")
    print(f" {'Carry Drop':>12}")
    print("-" * 74)

    print(f"{'(none - baseline)':<20}", end="")
    for cat in categories:
        print(f" {baseline[cat]['accuracy']:>10.1%}", end="")
    print(f" {'--':>12}")

    def carry_drop(res):
        ba_drop = baseline["ba"]["accuracy"] - res["ba"]["accuracy"]
        carry_drops = [
            baseline[c]["accuracy"] - res[c]["accuracy"]
            for c in ["mc1", "us9", "s3"]
        ]
        return np.mean(carry_drops) - ba_drop

    in_circuit_set = set(_comp_label(c) for c in circuit_flat)

    for comp in all_comps:
        label = _comp_label(comp)
        res = results["individual"][label]
        marker = " *" if label in in_circuit_set else ""
        print(f"{label + marker:<20}", end="")
        for cat in categories:
            print(f" {res[cat]['accuracy']:>10.1%}", end="")
        cd = carry_drop(res)
        print(f" {cd:>+12.1%}")

    print("-" * 74)
    for group_name, group_label in [("all_circuit", "ALL circuit"),
                                     ("all_non_circuit", "ALL non-circuit")]:
        res = results["group"][group_name]
        print(f"{group_label:<20}", end="")
        for cat in categories:
            print(f" {res[cat]['accuracy']:>10.1%}", end="")
        cd = carry_drop(res)
        print(f" {cd:>+12.1%}")

    circuit_cd = carry_drop(results["group"]["all_circuit"])
    non_circuit_cd = carry_drop(results["group"]["all_non_circuit"])
    verdict = "PASS" if abs(circuit_cd) > abs(non_circuit_cd) * 2 else "FAIL"
    print(f"\nCircuit carry drop: {circuit_cd:+.1%}, Non-circuit carry drop: {non_circuit_cd:+.1%}")
    print(f"Necessity verdict: {verdict}")

    results["verdict"] = verdict
    return results


def test_sufficiency(model, test_data, circuit_flat, non_circuit, mean_acts,
                     n_random=100):
    print("\n" + "=" * 60)
    print("SUFFICIENCY TESTING")
    print("=" * 60)

    print("Computing baseline accuracy...")
    baseline = _ablate_and_evaluate(model, test_data, mean_acts, [])

    print("Computing circuit-only accuracy (ablating non-circuit)...")
    circuit_only = _ablate_and_evaluate(model, test_data, mean_acts, non_circuit)

    print(f"Computing {n_random} random circuit baselines...")
    all_comps = circuit_flat + non_circuit
    circuit_size = len(circuit_flat)
    random.seed(SEED)

    random_recoveries = {cat: [] for cat in ["ba", "mc1", "us9", "s3"]}

    for i in range(n_random):
        if (i + 1) % 20 == 0:
            print(f"  Random circuit {i+1}/{n_random}...")
        rand_circuit = random.sample(all_comps, circuit_size)
        rand_non_circuit = [c for c in all_comps if c not in rand_circuit]
        rand_result = _ablate_and_evaluate(model, test_data, mean_acts, rand_non_circuit)
        for cat in random_recoveries:
            base_acc = baseline[cat]["accuracy"]
            if base_acc > 0:
                random_recoveries[cat].append(rand_result[cat]["accuracy"] / base_acc * 100)
            else:
                random_recoveries[cat].append(0.0)

    categories = ["ba", "mc1", "us9", "s3"]
    cat_display = {"ba": "BA", "mc1": "MC1", "us9": "US9", "s3": "S3"}

    print(f"\n{'Category':<12} {'Full Model':>12} {'Circuit Only':>14} {'Recovery':>10} "
          f"{'Rand Mean+-Std':>16} {'Percentile':>12}")
    print("-" * 60)

    results = {
        "baseline": baseline,
        "circuit_only": circuit_only,
        "categories": {},
        "random_recoveries": {},
    }

    for cat in categories:
        base_acc = baseline[cat]["accuracy"]
        circ_acc = circuit_only[cat]["accuracy"]
        recovery = (circ_acc / base_acc * 100) if base_acc > 0 else 0

        rand_vals = random_recoveries[cat]
        rand_mean = np.mean(rand_vals)
        rand_std = np.std(rand_vals)

        percentile = np.sum(np.array(rand_vals) < recovery) / len(rand_vals) * 100

        print(f"{cat_display[cat]:<12} {base_acc:>12.1%} {circ_acc:>14.1%} "
              f"{recovery:>9.1f}% {rand_mean:>8.1f}% +- {rand_std:<5.1f}% "
              f"{percentile:>10.0f}th")

        results["categories"][cat] = {
            "full_model_acc": base_acc,
            "circuit_only_acc": circ_acc,
            "recovery_pct": recovery,
            "random_mean": rand_mean,
            "random_std": rand_std,
            "percentile": percentile,
        }
        results["random_recoveries"][cat] = rand_vals

    avg_recovery = np.mean([results["categories"][c]["recovery_pct"] for c in categories])
    avg_percentile = np.mean([results["categories"][c]["percentile"] for c in categories])
    verdict = "PASS" if avg_recovery > 80 and avg_percentile > 90 else "FAIL"
    print(f"\nMean recovery: {avg_recovery:.1f}%, Mean percentile: {avg_percentile:.0f}th")
    print(f"Sufficiency verdict: {verdict}")

    results["verdict"] = verdict
    return results


def detect_compensation(model, pairs, circuit_flat, non_circuit, mean_acts):
    print("\n" + "=" * 60)
    print("COMPENSATORY PATH DETECTION")
    print("=" * 60)

    pair_list = pairs["type_a_ones"]
    primary_ai = pair_list[0]["primary_answer_idx"]

    primary_circuit = [(l, h) for l, h in circuit_flat if isinstance(h, int) and l == 0]

    def compute_head_ie(pair_list, primary_ai, ablation_hooks=None):
        if ablation_hooks is None:
            ablation_hooks = []

        from patching import compute_logit_diff, compute_ie_stolfo

        head_ies = {}
        n = len(pair_list)
        batch_size = 64

        for start in range(0, n, batch_size):
            batch = pair_list[start:start + batch_size]
            cl_tok = torch.tensor([p["clean_tokens"] for p in batch], dtype=torch.long)
            co_tok = torch.tensor([p["corrupted_tokens"] for p in batch], dtype=torch.long)
            cl_in = cl_tok[:, :-1]
            co_in = co_tok[:, :-1]

            lp = ANSWER_LOGIT_POS[primary_ai]
            ct = cl_tok[:, ANSWER_SEQ_POS[primary_ai]]
            rt = co_tok[:, ANSWER_SEQ_POS[primary_ai]]

            with torch.no_grad():
                clean_head_acts = {}
                def make_cache_hook(store_key):
                    def hook_fn(activation, **kwargs):
                        clean_head_acts[store_key] = activation.clone()
                        return activation
                    return hook_fn

                cache_hooks = []
                for l in range(N_LAYERS):
                    hname = f"blocks.{l}.attn.hook_result"
                    cache_hooks.append((hname, make_cache_hook(hname)))

                cl_logits = model.run_with_hooks(
                    cl_in, fwd_hooks=ablation_hooks + cache_hooks)

                co_logits = model.run_with_hooks(
                    co_in, fwd_hooks=ablation_hooks)

                cl_ld = compute_logit_diff(cl_logits, ct, rt, lp)
                co_ld = compute_logit_diff(co_logits, ct, rt, lp)

                for l in range(N_LAYERS):
                    for h in range(N_HEADS):
                        label = f"L{l}H{h}"
                        hook_name = f"blocks.{l}.attn.hook_result"
                        cl_act = clean_head_acts[hook_name]

                        def make_patch_hook(cl_a, hi):
                            def hook_fn(activation, **kwargs):
                                activation[:, :, hi, :] = cl_a[:, :, hi, :]
                                return activation
                            return hook_fn

                        patched_logits = model.run_with_hooks(
                            co_in,
                            fwd_hooks=ablation_hooks + [
                                (hook_name, make_patch_hook(cl_act, h))
                            ])

                        pa_ld = compute_logit_diff(patched_logits, ct, rt, lp)
                        ie = compute_ie_stolfo(cl_ld, pa_ld, co_ld).mean().item()

                        if label not in head_ies:
                            head_ies[label] = []
                        head_ies[label].append(ie)

        return {k: np.mean(v) for k, v in head_ies.items()}

    print("\nComputing baseline IE for all heads...")
    baseline_ie = compute_head_ie(pair_list, primary_ai)

    print("Computing IE after ablating primary circuit...")
    ablation_hooks = []
    for l, h in primary_circuit:
        hook_name = f"blocks.{l}.attn.hook_result"
        mean_val = mean_acts[hook_name]

        def make_ablation_hook(mean_v, hi):
            def hook_fn(activation, **kwargs):
                activation[:, :, hi, :] = mean_v[:, hi, :].unsqueeze(0)
                return activation
            return hook_fn

        ablation_hooks.append((hook_name, make_ablation_hook(mean_val, h)))

    ablated_ie = compute_head_ie(pair_list, primary_ai, ablation_hooks=ablation_hooks)

    print("Identifying compensatory paths...\n")
    non_circuit_labels = set(_comp_label(c) for c in non_circuit
                             if isinstance(c[1], int))

    results = {"baseline_ie": baseline_ie, "ablated_ie": ablated_ie, "compensators": []}

    print(f"{'Component':<12} {'IE (baseline)':>14} {'IE (ablated)':>14} "
          f"{'Change':>10} {'Classification'}")
    print("-" * 72)

    for label in HEAD_LABELS:
        b_ie = baseline_ie.get(label, 0)
        a_ie = ablated_ie.get(label, 0)
        if abs(b_ie) > 1e-6:
            change_pct = (a_ie - b_ie) / abs(b_ie) * 100
        else:
            change_pct = 0 if abs(a_ie) < 1e-6 else float('inf')

        is_non_circuit = label in non_circuit_labels
        is_compensator = is_non_circuit and (
            (abs(b_ie) > 0.01 and change_pct > 50) or
            (abs(a_ie) > 0.1)
        )

        if is_compensator:
            classification = "Active compensation"
            results["compensators"].append({
                "component": label,
                "baseline_ie": b_ie,
                "ablated_ie": a_ie,
                "change_pct": change_pct,
                "classification": classification,
            })
        elif is_non_circuit:
            classification = "Not compensating"
        else:
            classification = "(in circuit)"

        marker = " *" if is_compensator else ""
        print(f"{label:<12} {b_ie:>14.4f} {a_ie:>14.4f} "
              f"{change_pct:>+9.0f}% {classification}{marker}")

    if results["compensators"]:
        print(f"\nActive compensators found: "
              f"{[c['component'] for c in results['compensators']]}")
    else:
        print("\nNo active compensators found.")

    return results


def test_minimality(model, test_data, circuit_flat, non_circuit, mean_acts):
    # for each circuit component, remove it and check if recovery drops
    print("\n" + "=" * 60)
    print("MINIMALITY TESTING")
    print("=" * 60)

    baseline = _ablate_and_evaluate(model, test_data, mean_acts, [])

    full_circuit = _ablate_and_evaluate(model, test_data, mean_acts, non_circuit)

    categories = ["ba", "mc1", "us9", "s3"]
    cat_display = {"ba": "BA", "mc1": "MC1", "us9": "US9", "s3": "S3"}

    def mean_recovery(res):
        recoveries = []
        for cat in categories:
            base = baseline[cat]["accuracy"]
            if base > 0:
                recoveries.append(res[cat]["accuracy"] / base * 100)
        return np.mean(recoveries)

    full_circuit_recovery = mean_recovery(full_circuit)
    print(f"\nFull circuit recovery: {full_circuit_recovery:.1f}%")

    results = {
        "full_circuit_recovery": full_circuit_recovery,
        "components": {},
        "redundant": [],
    }

    print(f"\n{'Removed Component':<20} {'Recovery':>10} {'Drop':>10} {'Status'}")
    print("-" * 56)

    for comp in circuit_flat:
        label = _comp_label(comp)
        reduced_circuit = [c for c in circuit_flat if c != comp]
        reduced_non_circuit = [c for c in (circuit_flat + non_circuit) if c not in reduced_circuit]

        res = _ablate_and_evaluate(model, test_data, mean_acts, reduced_non_circuit)
        recovery = mean_recovery(res)
        drop = full_circuit_recovery - recovery

        is_redundant = drop < 1.0
        status = "REDUNDANT" if is_redundant else "ESSENTIAL"

        print(f"{label:<20} {recovery:>9.1f}% {drop:>+9.1f}% {status}")

        results["components"][label] = {
            "recovery": recovery,
            "drop": drop,
            "status": status,
        }
        if is_redundant:
            results["redundant"].append(label)

    if results["redundant"]:
        print(f"\nRedundant components: {results['redundant']}")
    else:
        print("\nNo redundant components found -- circuit is minimal.")

    return results
