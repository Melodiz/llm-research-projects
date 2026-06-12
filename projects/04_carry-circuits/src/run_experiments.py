# run all patching experiments

import os
import sys
import time
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from config import (
    N_LAYERS, N_HEADS, D_MODEL, D_HEAD, D_MLP, VOCAB_SIZE,
    SEQ_LEN, ACT_FN, NORMALIZATION_TYPE, DEVICE, SEED,
)
from counterfactual import generate_all_pairs, load_pairs
from experiments import (
    experiment_1, experiment_2, experiment_3, experiment_4, experiment_5,
    RESULTS_DIR,
)

def load_model(checkpoint_path):
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        d_vocab=VOCAB_SIZE,
        n_ctx=SEQ_LEN,
        act_fn=ACT_FN,
        normalization_type=NORMALIZATION_TYPE,
        device=DEVICE,
        seed=SEED,
        use_attn_result=True,  # needed for head-level patching via hook_result
    )
    model = HookedTransformer(cfg)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def main():
    torch.manual_seed(SEED)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ckpt = os.path.join(base_dir, "checkpoints", "best_model.pt")
    model = load_model(ckpt)

    pairs_path = os.path.join(RESULTS_DIR, "counterfactual_pairs.json")
    if os.path.exists(pairs_path):
        print("\nLoading existing counterfactual pairs...")
        pairs = load_pairs(RESULTS_DIR)
        for k, v in pairs.items():
            print(f"  {k}: {len(v)} pairs")
    else:
        print("\nGenerating counterfactual pairs...")
        pairs = generate_all_pairs(n_pairs_per_type=500, save_dir=RESULTS_DIR)

    # run experiments
    summaries = {}
    t0 = time.time()

    print("\n" + "=" * 60)
    print("  RUNNING EXPERIMENTS")
    print("=" * 60)

    s = experiment_1(model, pairs)
    summaries["Exp 1 (Residual stream)"] = s

    s = experiment_2(model, pairs)
    summaries["Exp 2 (Components)"] = s

    s = experiment_3(model, pairs)
    summaries["Exp 3 (Heads)"] = s

    s = experiment_4(model)
    summaries["Exp 4 (Attention)"] = s

    s = experiment_5(model, pairs)
    summaries["Exp 5 (Logit attribution)"] = s
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)

    for name, findings in summaries.items():
        print(f"\n{name}:")
        for f in findings:
            print(f"  {f}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}/")

    print("\nOutput files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
