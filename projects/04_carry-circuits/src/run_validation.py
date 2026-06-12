# runs all circuit validation experiments
import os, sys, json, time
import torch

from config import SEED, DEVICE
from run_experiments import load_model
from data import load_test_sets
from counterfactual import load_pairs
from circuit import (
    define_circuit_from_results,
    compute_mean_activations,
    test_necessity,
    test_sufficiency,
    detect_compensation,
    test_minimality,
    _comp_label, _comp_labels,
    RESULTS_DIR, VALIDATION_DIR,
)


def _save_json(name, data):
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    path = os.path.join(VALIDATION_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path

def main():
    torch.manual_seed(SEED)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    t0 = time.time()

    ckpt = os.path.join(base_dir, "checkpoints", "best_model.pt")
    model = load_model(ckpt)

    test_dir = os.path.join(base_dir, "data")
    test_data = load_test_sets(test_dir)

    pairs = load_pairs(RESULTS_DIR)

    # circuit definition
    circuit_def, circuit_flat, non_circuit, ranking = define_circuit_from_results()

    circuit_info = {
        "circuit": {
            "primary": [_comp_label(c) for c in circuit_def["primary"]],
            "secondary": [_comp_label(c) for c in circuit_def["secondary"]],
            "mlp": [_comp_label(c) for c in circuit_def["mlp"]],
        },
        "circuit_flat": _comp_labels(circuit_flat),
        "non_circuit": _comp_labels(non_circuit),
        "ranking": ranking,
    }
    _save_json("circuit_definition.json", circuit_info)

    # mean activations
    print("\nComputing per-position mean activations over 2000 diverse examples...")
    mean_acts = compute_mean_activations(model, n_examples=2000)
    print("  Done.")

    # necessity
    necessity_results = test_necessity(model, test_data, circuit_flat, non_circuit, mean_acts)
    _save_json("necessity.json", {
        k: v for k, v in necessity_results.items()
        if k != "baseline"
    })

    # sufficiency
    sufficiency_results = test_sufficiency(
        model, test_data, circuit_flat, non_circuit, mean_acts, n_random=100)
    suff_save = {k: v for k, v in sufficiency_results.items()
                 if k != "random_recoveries"}
    suff_save["random_recoveries_summary"] = {
        cat: {"values": vals}
        for cat, vals in sufficiency_results.get("random_recoveries", {}).items()
    }
    _save_json("sufficiency.json", suff_save)

    # compensation
    compensation_results = detect_compensation(
        model, pairs, circuit_flat, non_circuit, mean_acts)
    _save_json("compensation.json", compensation_results)

    # minimality
    minimality_results = test_minimality(
        model, test_data, circuit_flat, non_circuit, mean_acts)
    _save_json("minimality.json", minimality_results)

    elapsed = time.time() - t0

    # report
    print("\n")
    print("=" * 50)
    print("CIRCUIT VALIDATION REPORT")
    print("=" * 50)

    print(f"\n1. CIRCUIT DEFINITION")
    print(f"   Primary:   {circuit_info['circuit']['primary']}")
    print(f"   Secondary: {circuit_info['circuit']['secondary']}")
    print(f"   MLP:       {circuit_info['circuit']['mlp']}")
    print(f"   Size: {len(circuit_flat)} of 8 total components "
          f"({100*len(circuit_flat)/8:.0f}%)")

    print(f"\n2. NECESSITY")
    print(f"   Verdict: {necessity_results['verdict']}")

    print(f"\n3. SUFFICIENCY")
    for cat in ["ba", "mc1", "us9", "s3"]:
        cat_data = sufficiency_results["categories"][cat]
        print(f"   {cat.upper()}: {cat_data['recovery_pct']:.1f}% recovery "
              f"({cat_data['percentile']:.0f}th percentile)")
    print(f"   Verdict: {sufficiency_results['verdict']}")

    print(f"\n4. COMPENSATORY PATHS")
    comps = compensation_results.get("compensators", [])
    if comps:
        print(f"   Active compensators: {[c['component'] for c in comps]}")
    else:
        print(f"   No active compensators found.")

    print(f"\n5. MINIMALITY")
    redundant = minimality_results.get("redundant", [])
    if redundant:
        print(f"   Redundant components: {redundant}")
    else:
        print(f"   All components are essential (minimal circuit).")

    suff_pass = sufficiency_results["verdict"] == "PASS"
    nec_pass = necessity_results["verdict"] == "PASS"
    is_minimal = len(redundant) == 0
    has_compensation = len(comps) > 0

    if nec_pass and suff_pass and is_minimal:
        overall = "LOCALIZED -- modular carry circuit"
    elif nec_pass and suff_pass:
        overall = "PARTIALLY LOCALIZED -- some redundancy"
    elif nec_pass:
        overall = "NECESSARY but not sufficient -- distributed processing"
    else:
        overall = "DISTRIBUTED -- carry computation not strongly localized"

    print(f"\n6. OVERALL VERDICT")
    print(f"   Circuit is: {overall}")
    print(f"   Carry computation is: "
          f"{'modular' if nec_pass and suff_pass else 'partially distributed'}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {VALIDATION_DIR}/")

    # print output files
    print("\nOutput files:")
    for f in sorted(os.listdir(VALIDATION_DIR)):
        fpath = os.path.join(VALIDATION_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
