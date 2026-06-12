"""Reward functions for edge counting positive control."""

import re
from edge_counting_env import EdgeCountingVerifier, Data

_FMT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>.*?</answer>",
    re.DOTALL | re.IGNORECASE,
)

_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_text(completion):
    """str or trl >=0.16 list-of-dicts → plain text."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return completion or ""


def _check_correct(text, expected):
    """Check if the completion contains the correct edge count."""
    # Try <answer> tags first
    matches = list(_ANSWER_TAG_RE.finditer(text))
    if matches:
        answer_text = matches[-1].group(1).strip()
        nums = re.findall(r'\d+', answer_text)
        if nums:
            return int(nums[0]) == expected
        return False

    # Fallback: find last integer in raw text
    nums = re.findall(r'\d+', text)
    if nums:
        return int(nums[-1]) == expected
    return False


def ec_reward(completions, edge_count, **kwargs):
    """Combined edge counting reward for GRPOTrainer."""
    rewards = []

    for comp, ec in zip(completions, edge_count):
        text = _extract_text(comp)
        expected = int(ec)

        # Format component: +0.5 if proper tags, 0.0 otherwise
        fmt = 0.5 if _FMT_RE.search(text) else 0.0

        # Correctness component: +1.0 if correct, -1.0 if wrong
        correct = 1.0 if _check_correct(text, expected) else -1.0

        reward = 0.7 * correct + 0.3 * fmt
        rewards.append(reward)

    return rewards
