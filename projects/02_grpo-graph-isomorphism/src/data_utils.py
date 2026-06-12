"""Data preparation utilities for Graph Isomorphism GRPO training."""
import random
from datasets import Dataset
from graph_isomorphism_env import GraphIsomorphismEnv, DIFFICULTY_MAP


def _resolve_hints_k(hints_k_raw, difficulty):
    """Resolve hints_k."""
    if hints_k_raw is None:
        return None
    if isinstance(hints_k_raw, int):
        return hints_k_raw
    # String expressions like "n-2"
    s = str(hints_k_raw).strip()
    if s.isdigit():
        return int(s)
    n = DIFFICULTY_MAP.get(difficulty, DIFFICULTY_MAP[5])["n_nodes"]
    try:
        return max(0, eval(s, {"n": n, "__builtins__": {}}))
    except Exception:
        raise ValueError(f"Cannot resolve hints_k={hints_k_raw!r} (n={n})")


def prepare_dataset(difficulty=3, iso_ratio=0.5, num_questions=500, seed=42,
                    config=None, **kwargs) -> Dataset:
    """Prepares the HuggingFace Dataset for GRPO training."""
    hints_k_raw = None
    # Config dict overrides defaults (for train.py compatibility)
    if config is not None:
        difficulty = config.get("difficulty", difficulty)
        iso_ratio = config.get("iso_ratio", iso_ratio)
        num_questions = config.get("num_questions", num_questions)
        seed = config.get("seed", seed)
        hints_k_raw = config.get("hints_k", None)
    # kwargs override everything
    difficulty = kwargs.get("difficulty", difficulty)
    iso_ratio = kwargs.get("iso_ratio", iso_ratio)
    num_questions = kwargs.get("num_questions", num_questions)
    seed = kwargs.get("seed", seed)
    hints_k_raw = kwargs.get("hints_k", hints_k_raw)

    hints_k = _resolve_hints_k(hints_k_raw, difficulty)

    env = GraphIsomorphismEnv()

    num_iso = int(num_questions * iso_ratio)
    num_non_iso = num_questions - num_iso

    dataset = []

    SYSTEM_PROMPT = (
        "Respond in the following format:\n"
        "<think>\n...\n</think>\n"
        "<answer>\n...\n</answer>"
    )

    def _convert(data_list, ttype):
        for d in data_list:
            meta = dict(d.metadata)
            if "ground_truth_perm" in meta:
                meta["ground_truth_perm"] = {
                    str(k): v for k, v in meta["ground_truth_perm"].items()
                }
            dataset.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d.question},
                ],
                "ground_truth": d.answer,
                "metadata_list": meta,
                "task_type": ttype,
            })

    if num_iso > 0:
        iso_data = env.generate(
            num_of_questions=num_iso, difficulty=difficulty,
            iso_ratio=1.0, seed=42, hints_k=hints_k,
        )
        _convert(iso_data, "isomorphic")
        
    if num_non_iso > 0:
        non_iso_data = env.generate(
            num_of_questions=num_non_iso, difficulty=difficulty,
            iso_ratio=0.0, seed=43,
        )
        _convert(non_iso_data, "non_isomorphic")

    random.seed(seed)
    random.shuffle(dataset)
    return Dataset.from_list(dataset)
