# data generation and tokenization for addition task
import os
import json
import random

import torch
from torch.utils.data import Dataset
from config import (
    N_DIGITS, CHAR_TO_ID, EOS_TOKEN, SEQ_LEN, ANSWER_START,
    TRAIN_SIZE, TEST_SIZE_PER_CATEGORY, SEED
)


def classify_example(a, b, n_digits=N_DIGITS):
    # figure out carry pattern for a+b
    carry_pos = []
    carry = 0
    for pos in range(n_digits):
        d_a = (a // (10 ** pos)) % 10
        d_b = (b // (10 ** pos)) % 10
        col_sum = d_a + d_b + carry
        if col_sum >= 10:
            carry_pos.append(pos)
            carry = 1
        else:
            carry = 0
    if carry:
        carry_pos.append(n_digits)

    max_chain = count_carry_chain(a, b, n_digits)

    if len(carry_pos) == 0:
        category = "BA"
    elif max_chain <= 1:
        category = "MC1"
    else:
        category = "US9"

    return {
        "carry_positions": carry_pos,
        "category": category,
        "max_chain_length": max_chain,
    }

def count_carry_chain(a, b, n_digits=N_DIGITS):
    # longest consecutive carry chain
    max_chain = 0
    cur_chain = 0
    carry = 0
    for pos in range(n_digits):
        d_a = (a // (10 ** pos)) % 10
        d_b = (b // (10 ** pos)) % 10
        col_sum = d_a + d_b + carry
        if col_sum >= 10:
            cur_chain += 1
            carry = 1
        else:
            max_chain = max(max_chain, cur_chain)
            cur_chain = 0
            carry = 0
    if carry:
        cur_chain += 1
    max_chain = max(max_chain, cur_chain)
    return max_chain


def encode_example(a, b, n_digits=N_DIGITS):
    # turns a+b into token tensor
    s = a + b
    a_str = str(a).zfill(n_digits)
    b_str = str(b).zfill(n_digits)
    s_str = str(s).zfill(n_digits + 1)
    seq_str = a_str + '+' + b_str + '=' + s_str
    tokens = [CHAR_TO_ID[c] for c in seq_str] + [EOS_TOKEN]
    return torch.tensor(tokens, dtype=torch.long)

def generate_balanced_dataset(n_examples, n_digits=N_DIGITS):
    # sample uniformly across carry chain lengths
    max_val = 10 ** n_digits
    chain_lengths = list(range(n_digits + 1))  # 0, 1, 2, 3
    per_bucket = n_examples // len(chain_lengths)

    results = []
    for target_chain in chain_lengths:
        count = 0
        attempts = 0
        while count < per_bucket and attempts < per_bucket * 200:
            attempts += 1
            a = random.randint(0, max_val - 1)
            b = random.randint(0, max_val - 1)
            chain = count_carry_chain(a, b, n_digits)
            if chain == target_chain:
                info = classify_example(a, b, n_digits)
                results.append((a, b, info))
                count += 1

    # fill remainder
    while len(results) < n_examples:
        a = random.randint(0, max_val - 1)
        b = random.randint(0, max_val - 1)
        info = classify_example(a, b, n_digits)
        results.append((a, b, info))

    random.shuffle(results)
    return results[:n_examples]


def _generate_category_examples(category, n_examples, n_digits=N_DIGITS):
    max_val = 10 ** n_digits
    results = []
    while len(results) < n_examples:
        a = random.randint(0, max_val - 1)
        b = random.randint(0, max_val - 1)
        info = classify_example(a, b, n_digits)
        if category == "S3":
            if info["max_chain_length"] >= n_digits:
                results.append((a, b, info))
        elif info["category"] == category:
            results.append((a, b, info))
    return results

class AdditionDataset(Dataset):
    # TODO maybe add caching later
    def __init__(self, examples, n_digits=N_DIGITS):
        self.examples = examples
        self.n_digits = n_digits

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        a, b, info = self.examples[idx]
        tokens = encode_example(a, b, self.n_digits)

        input_tokens = tokens[:-1]   # length 12
        target_tokens = tokens[1:]   # length 12

        # loss only on answer positions
        loss_mask = torch.zeros(SEQ_LEN - 1, dtype=torch.float32)
        loss_mask[ANSWER_START - 1:] = 1.0

        return input_tokens, target_tokens, loss_mask, info


def generate_and_save_test_sets(save_dir, n_digits=N_DIGITS):
    os.makedirs(save_dir, exist_ok=True)

    n = TEST_SIZE_PER_CATEGORY
    max_val = 10 ** n_digits

    print("Generating test sets...")

    # 10k uniform random
    print("  Uniform random...")
    uniform = []
    for _ in range(n):
        a = random.randint(0, max_val - 1)
        b = random.randint(0, max_val - 1)
        info = classify_example(a, b, n_digits)
        uniform.append((a, b, info))

    for cat in ["BA", "MC1", "US9", "S3"]:
        print(f"  {cat}...")
        examples = _generate_category_examples(cat, n, n_digits)
        _save_examples(os.path.join(save_dir, f"test_{cat.lower()}.json"), examples)

    _save_examples(os.path.join(save_dir, "test_uniform.json"), uniform)
    print("Test sets saved.")

def _save_examples(path, examples):
    data = [(a, b, info) for a, b, info in examples]
    with open(path, 'w') as f:
        json.dump(data, f)

def _load_examples(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return [(a, b, info) for a, b, info in data]


def load_test_sets(save_dir):
    # not sure if the ordering matters here
    test_sets = {}
    for name in ["uniform", "ba", "mc1", "us9", "s3"]:
        path = os.path.join(save_dir, f"test_{name}.json")
        test_sets[name] = _load_examples(path)
    return test_sets


if __name__ == "__main__":
    random.seed(SEED)
    print("Example encoding:")
    tokens = encode_example(12, 34)
    print(f"  12 + 34 = 46 -> tokens: {tokens.tolist()}")

    info = classify_example(12, 34)
    print(f"  Classification: {info}")

    info2 = classify_example(999, 1)
    print(f"  999 + 1: {info2}")

    info3 = classify_example(195, 805)
    print(f"  195 + 805: {info3}")

    # print("\nGenerating small balanced dataset...")
    print("\nBalanced dataset test...")
    examples = generate_balanced_dataset(1000)
    cats = {}
    for _, _, info in examples:
        cat = info["category"]
        cats[cat] = cats.get(cat, 0) + 1
    print(f"  Category distribution: {cats}")
