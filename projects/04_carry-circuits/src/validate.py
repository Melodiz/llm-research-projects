# evaluation helpers
import torch
from config import ANSWER_START, SEQ_LEN, N_DIGITS, EOS_TOKEN, ID_TO_CHAR
from data import encode_example, AdditionDataset

def evaluate_model(model, test_data, batch_size=256):
    model.eval()
    results = {}

    all_examples = []
    all_categories = []
    for cat_name, examples in test_data.items():
        all_examples.extend(examples)
        all_categories.extend([cat_name] * len(examples))

    n = len(all_examples)
    correct_full = []
    correct_per_digit = [[] for _ in range(N_DIGITS + 1)]  # 4 answer digits
    categories_list = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_examples = all_examples[i:i + batch_size]
            batch_cats = all_categories[i:i + batch_size]

            tokens_list = []
            for a, b, info in batch_examples:
                tokens_list.append(encode_example(a, b))
            tokens = torch.stack(tokens_list)  # [B, 13]

            input_tokens = tokens[:, :-1]  # [B, 12]
            target_tokens = tokens[:, 1:]  # [B, 12]

            logits = model(input_tokens)  # [B, 12, vocab]
            preds = logits.argmax(dim=-1)  # [B, 12]

            answer_preds = preds[:, ANSWER_START - 1:]  # [B, 5] (4 digits + EOS)
            answer_targets = target_tokens[:, ANSWER_START - 1:]  # [B, 5]

            # not sure if ignoring EOS here is correct but it seems to work
            digit_preds = answer_preds[:, :N_DIGITS + 1]  # [B, 4]
            digit_targets = answer_targets[:, :N_DIGITS + 1]  # [B, 4]

            full_correct = (digit_preds == digit_targets).all(dim=1)  # [B]
            correct_full.extend(full_correct.tolist())

            for d in range(N_DIGITS + 1):
                d_correct = (digit_preds[:, d] == digit_targets[:, d]).tolist()
                correct_per_digit[d].extend(d_correct)

            categories_list.extend(batch_cats)

    results["overall_accuracy"] = sum(correct_full) / len(correct_full)

    cat_correct = {}
    cat_total = {}
    for i, cat in enumerate(categories_list):
        cat_correct[cat] = cat_correct.get(cat, 0) + correct_full[i]
        cat_total[cat] = cat_total.get(cat, 0) + 1
    results["per_category"] = {
        cat: cat_correct[cat] / cat_total[cat] for cat in sorted(cat_total.keys())
    }

    digit_names = ["thousands(overflow)", "hundreds", "tens", "ones"]
    results["per_digit"] = {}
    for d in range(N_DIGITS + 1):
        results["per_digit"][digit_names[d]] = sum(correct_per_digit[d]) / len(correct_per_digit[d])

    # cross-tab: per-digit x per-category
    cross = {}
    for d in range(N_DIGITS + 1):
        cross[digit_names[d]] = {}
        cat_d_correct = {}
        cat_d_total = {}
        for i, cat in enumerate(categories_list):
            cat_d_correct[cat] = cat_d_correct.get(cat, 0) + correct_per_digit[d][i]
            cat_d_total[cat] = cat_d_total.get(cat, 0) + 1
        for cat in sorted(cat_d_total.keys()):
            cross[digit_names[d]][cat] = cat_d_correct[cat] / cat_d_total[cat]
    results["cross_tab"] = cross

    return results


def print_results(results, step=None):
    # print summary table
    header = f"Step {step}" if step is not None else "Final"
    print(f"\n{'=' * 70}")
    print(f"  Evaluation Results — {header}")
    print(f"{'=' * 70}")

    print(f"\n  Overall accuracy: {results['overall_accuracy']:.4f}")

    print(f"\n  Per-category accuracy:")
    print(f"  {'Category':<12} {'Accuracy':>10}")
    print(f"  {'-' * 22}")
    for cat, acc in sorted(results["per_category"].items()):
        print(f"  {cat:<12} {acc:>10.4f}")

    print(f"\n  Per-digit accuracy:")
    print(f"  {'Digit':<20} {'Accuracy':>10}")
    print(f"  {'-' * 30}")
    for digit, acc in results["per_digit"].items():
        print(f"  {digit:<20} {acc:>10.4f}")

    print(f"\n  Cross-tabulation (digit × category):")
    cats = sorted(set().union(*[v.keys() for v in results["cross_tab"].values()]))
    header_line = f"  {'Digit':<20}" + "".join(f"{c:>10}" for c in cats)
    print(header_line)
    print(f"  {'-' * (20 + 10 * len(cats))}")
    for digit, cat_accs in results["cross_tab"].items():
        line = f"  {digit:<20}"
        for cat in cats:
            line += f"{cat_accs.get(cat, 0):.4f}".rjust(10)
        print(line)

    print(f"{'=' * 70}\n")
