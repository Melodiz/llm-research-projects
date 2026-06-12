# counterfactual pair generation for patching experiments
import os
import json
import random
import torch

from data import encode_example
from config import N_DIGITS, SEED

# answer digit index (0=overflow, 1=hundreds, 2=tens, 3=ones)
# maps to logit position (7, 8, 9, 10) and full-sequence position (8, 9, 10, 11)
ANSWER_LOGIT_POS = [7, 8, 9, 10]
ANSWER_SEQ_POS = [8, 9, 10, 11]

# carry at position k -> affects answer at this answer_idx
CARRY_TO_ANSWER_IDX = {0: 2, 1: 1, 2: 0}  # ones->tens, tens->hundreds, hundreds->overflow

def _get_digit(n, pos):
    return (n // (10 ** pos)) % 10

def _set_digit(n, pos, d):
    return n - _get_digit(n, pos) * (10 ** pos) + d * (10 ** pos)


def _make_pair(a_clean, b_clean, a_corr, b_corr, target_k, pair_type, primary_ans_idx):
    clean_tok = encode_example(a_clean, b_clean).tolist()
    corr_tok = encode_example(a_corr, b_corr).tolist()
    return {
        "clean_tokens": clean_tok,
        "corrupted_tokens": corr_tok,
        "a_clean": a_clean, "b_clean": b_clean,
        "a_corrupted": a_corr, "b_corrupted": b_corr,
        "target_carry_pos": target_k,
        "primary_answer_idx": primary_ans_idx,
        "primary_logit_pos": ANSWER_LOGIT_POS[primary_ans_idx],
        "type": pair_type,
    }


def generate_type_a_pairs(target_k, n_pairs=500):
    # type A: single carry isolation at position k
    # clean has carry at k, corrupted does not
    pairs = []
    max_val = 10 ** N_DIGITS

    while len(pairs) < n_pairs:
        a = random.randint(0, max_val - 1)
        b = random.randint(0, max_val - 1)

        valid = True
        carry = 0
        for pos in range(N_DIGITS):
            da = _get_digit(a, pos)
            db = _get_digit(b, pos)
            s = da + db + carry
            if pos < target_k:
                if s >= 10:
                    valid = False
                    break
                carry = 0
            elif pos == target_k:
                if s < 10:
                    valid = False
                    break
                carry = 1
            else:
                if s >= 10:
                    valid = False
                    break
                carry = 0
        if not valid:
            continue

        db_k = _get_digit(b, target_k)
        # need new_da + db_k < 10 (carry_in at k is 0 since no carries below)
        max_da = 9 - db_k
        if max_da < 0:
            continue
        new_da = random.randint(0, max_da)
        a_corr = _set_digit(a, target_k, new_da)

        if a + b >= 10000 or a_corr + b >= 10000:
            continue

        ans_idx = CARRY_TO_ANSWER_IDX[target_k]
        pairs.append(_make_pair(a, b, a_corr, b, target_k, "A", ans_idx))

    return pairs

def generate_type_b_pairs(n_pairs=500):
    # type B: cascade isolation at tens
    # both have carry at ones, clean has tens sum=9 so carry cascades
    pairs = []

    while len(pairs) < n_pairs:
        ones_b = random.randint(1, 9)
        ones_a = random.randint(10 - ones_b, 9)

        tens_b = random.randint(0, 9)
        tens_a_clean = 9 - tens_b

        # corrupted: tens_a' + tens_b <= 7 so +1 carry doesnt overflow
        max_tens_corr = min(7 - tens_b, 9)
        if max_tens_corr < 0:
            continue
        tens_a_corr = random.randint(0, max_tens_corr)

        # hundreds: no carry either way
        hundreds_b = random.randint(0, 8)
        hundreds_a = random.randint(0, 8 - hundreds_b)

        a_clean = hundreds_a * 100 + tens_a_clean * 10 + ones_a
        b = hundreds_b * 100 + tens_b * 10 + ones_b
        a_corr = hundreds_a * 100 + tens_a_corr * 10 + ones_a

        if a_clean + b >= 10000 or a_corr + b >= 10000:
            continue

        # cascade affects hundreds answer -> answer_idx 1, logit_pos 8
        pairs.append(_make_pair(a_clean, b, a_corr, b, 1, "B", 1))

    return pairs


def generate_type_c_pairs(n_pairs=500):
    # type C: confound control - same ones answer digit, different carry status
    pairs = []

    while len(pairs) < n_pairs:
        ones_result = random.randint(0, 8)

        # clean: carry at ones
        lo = max(ones_result + 1, 1)
        if lo > 9:
            continue
        ones_a_cl = random.randint(lo, 9)
        ones_b_cl = ones_result + 10 - ones_a_cl
        if ones_b_cl < 0 or ones_b_cl > 9:
            continue

        # corrupted: no carry at ones, same result
        ones_a_co = random.randint(0, ones_result)
        ones_b_co = ones_result - ones_a_co

        # tens: same in both, no cascade
        tens_a = random.randint(0, 8)
        tens_b = random.randint(0, 8 - tens_a)

        hundreds_a = random.randint(0, 9)
        hundreds_b = random.randint(0, 9 - hundreds_a)

        a_cl = hundreds_a * 100 + tens_a * 10 + ones_a_cl
        b_cl = hundreds_b * 100 + tens_b * 10 + ones_b_cl
        a_co = hundreds_a * 100 + tens_a * 10 + ones_a_co
        b_co = hundreds_b * 100 + tens_b * 10 + ones_b_co

        if a_cl + b_cl >= 10000 or a_co + b_co >= 10000:
            continue

        # measure at tens answer (logit pos 9)
        pairs.append(_make_pair(a_cl, b_cl, a_co, b_co, 0, "C", 2))

    return pairs

# TODO maybe add type D pairs later?

def generate_all_pairs(n_pairs_per_type=500, save_dir=None):
    random.seed(SEED)

    all_pairs = {
        "type_a_ones": generate_type_a_pairs(0, n_pairs_per_type),
        "type_a_tens": generate_type_a_pairs(1, n_pairs_per_type),
        "type_a_hundreds": generate_type_a_pairs(2, n_pairs_per_type),
        "type_b": generate_type_b_pairs(n_pairs_per_type),
        "type_c": generate_type_c_pairs(n_pairs_per_type),
    }

    for name, p in all_pairs.items():
        print(f"  {name}: {len(p)} pairs")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "counterfactual_pairs.json")
        with open(path, 'w') as f:
            json.dump(all_pairs, f)
        print(f"  Saved to {path}")

    return all_pairs

def load_pairs(save_dir):
    path = os.path.join(save_dir, "counterfactual_pairs.json")
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    random.seed(SEED)
    pairs = generate_all_pairs(n_pairs_per_type=10)

    # sanity check
    for name, plist in pairs.items():
        for p in plist:
            ct = p["clean_tokens"]
            rt = p["corrupted_tokens"]
            idx = p["primary_answer_idx"]
            seq_pos = ANSWER_SEQ_POS[idx]
            assert ct[seq_pos] != rt[seq_pos], (
                f"{name}: answers same at primary pos {idx}: "
                f"clean={ct}, corrupted={rt}"
            )
    print("All sanity checks passed.")
