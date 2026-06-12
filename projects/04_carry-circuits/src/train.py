# training loop for addition transformer
import os, sys, json
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig

from config import (
    N_DIGITS, VOCAB_SIZE, SEQ_LEN, ANSWER_START,
    N_LAYERS, N_HEADS, D_MODEL, D_HEAD, D_MLP, ACT_FN, NORMALIZATION_TYPE,
    BATCH_SIZE, LR, LR_MIN, WEIGHT_DECAY, WARMUP_STEPS, TOTAL_STEPS,
    GRAD_CLIP_NORM, EVAL_EVERY, LOG_EVERY,
    TRAIN_SIZE, SEED, DEVICE
)
from data import (
    generate_balanced_dataset, generate_and_save_test_sets,
    load_test_sets, AdditionDataset
)
from validate import evaluate_model, print_results


def get_lr(step):
    # warmup then cosine decay
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    progress = min(progress, 1.0)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))

def collate_fn(batch):
    inputs, targets, masks, infos = zip(*batch)
    return (
        torch.stack(inputs),
        torch.stack(targets),
        torch.stack(masks),
        list(infos),
    )


def train():
    torch.manual_seed(SEED)
    random.seed(SEED)

    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(save_dir, "data")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(os.path.join(test_dir, "test_uniform.json")):
        random.seed(SEED + 1)
        generate_and_save_test_sets(test_dir)
        random.seed(SEED)

    test_data = load_test_sets(test_dir)
    print(f"Loaded test sets: {', '.join(f'{k}: {len(v)}' for k, v in test_data.items())}")

    print(f"Generating {TRAIN_SIZE} training examples...")
    train_examples = generate_balanced_dataset(TRAIN_SIZE)
    print(f"Training data generated. Category distribution:")
    cat_counts = {}
    for _, _, info in train_examples:
        cat = info["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} ({count/len(train_examples)*100:.1f}%)")

    train_dataset = AdditionDataset(train_examples)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True
    )

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
    )
    model = HookedTransformer(cfg)
    model.train()
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    training_log = []
    best_val_acc = 0.0
    step = 0
    epoch = 0
    data_iter = iter(train_loader)

    print(f"\nStarting training for {TOTAL_STEPS} steps...")
    print(f"{'Step':>6} {'Loss':>10} {'LR':>12}")
    print("-" * 30)

    while step < TOTAL_STEPS:
        try:
            input_tokens, target_tokens, loss_mask, _ = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            input_tokens, target_tokens, loss_mask, _ = next(data_iter)

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        logits = model(input_tokens)  # [B, seq_len-1, vocab]

        # cross-entropy on answer tokens only
        # logits: [B, 12, 14], target_tokens: [B, 12]
        loss_all = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target_tokens.reshape(-1),
            reduction='none'
        )  # [B * 12]
        loss_all = loss_all.reshape(target_tokens.shape)  # [B, 12]
        masked_loss = (loss_all * loss_mask).sum() / loss_mask.sum()

        optimizer.zero_grad()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        loss_val = masked_loss.item()

        if step % LOG_EVERY == 0:
            print(f"{step:>6} {loss_val:>10.6f} {lr:>12.6f}")
            training_log.append({
                "step": step,
                "train_loss": loss_val,
                "lr": lr,
            })

        if step > 0 and step % EVAL_EVERY == 0:
            model.eval()
            eval_results = evaluate_model(model, test_data)
            print_results(eval_results, step)
            model.train()

            training_log.append({
                "step": step,
                "eval": eval_results,
            })

            overall_acc = eval_results["overall_accuracy"]
            if overall_acc > best_val_acc:
                best_val_acc = overall_acc
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
                print(f"  New best model saved (accuracy: {overall_acc:.4f})")

        step += 1

    print("\n" + "=" * 70)
    print("  FINAL EVALUATION")
    print("=" * 70)
    model.eval()
    final_results = evaluate_model(model, test_data)
    print_results(final_results, step)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pt"))
    with open(os.path.join(save_dir, "results", "training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    print("Final model and training log saved.")


if __name__ == "__main__":
    train()
