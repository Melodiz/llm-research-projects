import re
from graph_isomorphism_env import GraphIsomorphismVerifier, Data

_FMT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>.*?</answer>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_text(completion):
    """Extract plain text from completion."""
    if isinstance(completion, list):
        # trl >=0.16: [[{"role": "assistant", "content": "..."}]]
        return completion[0]["content"] if completion else ""
    return completion or ""


# Payoff matrix (before +0.1 format bonus):
#
#              iso      non-iso
# correct     +1.5      +0.5
# wrong       -1.0      -0.5
#
# "Always NOT ISO" on 75/25 split: mean = 0.75*(-1.0) + 0.25*(+0.5) = -0.625
# => degenerate strategy is penalised

def class_aware_reward(completions, task_type, ground_truth, metadata_list, **kwargs):
    verifier = GraphIsomorphismVerifier()
    rewards = []
    for comp, tt, gt, meta in zip(completions, task_type, ground_truth, metadata_list):
        text = _extract_text(comp)
        data = Data(question="", answer=gt, metadata=meta)
        correct = verifier.verify(data, text)

        r = 0.1 if _FMT_RE.search(text) else 0.0

        if tt == "isomorphic":
            r += 1.5 if correct else -1.0
        else:
            r += 0.5 if correct else -0.5

        rewards.append(r)
    return rewards


# correct +1.0, wrong 0.0, +0.1 format bonus (vanilla control)

def composite_reward(completions, ground_truth, metadata_list, **kwargs):
    verifier = GraphIsomorphismVerifier()
    rewards = []
    for comp, gt, meta in zip(completions, ground_truth, metadata_list):
        text = _extract_text(comp)
        data = Data(question="", answer=gt, metadata=meta)

        r = 0.1 if _FMT_RE.search(text) else 0.0
        r += 1.0 if verifier.verify(data, text) else 0.0

        rewards.append(r)
    return rewards
