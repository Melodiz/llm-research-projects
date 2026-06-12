"""Edge Counting Environment for GRPO Training (Positive Control)."""

import re
import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from graph_isomorphism_env import Data, Verifier, Env, _graph_to_text



EC_DIFFICULTY_MAP = {
    1: {"n_nodes": 3, "edge_prob": 0.4},
    2: {"n_nodes": 4, "edge_prob": 0.4},
    3: {"n_nodes": 5, "edge_prob": 0.5},
}



class EdgeCountingVerifier(Verifier):
    """Verifier for edge counting."""

    ANSWER_PATTERN = re.compile(
        r"<answer>\s*(.*?)\s*</answer>",
        re.DOTALL | re.IGNORECASE,
    )

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution or not isinstance(test_solution, str):
            return ""
        matches = list(self.ANSWER_PATTERN.finditer(test_solution))
        if matches:
            return matches[-1].group(1).strip()
        return ""

    def verify(self, data: Data, test_solution: str) -> bool:
        answer_text = self.extract_answer(test_solution)
        if not answer_text:
            return False
        expected = data.metadata.get("edge_count")
        if expected is None:
            return False
        # Extract first integer from answer text
        nums = re.findall(r'\d+', answer_text)
        if not nums:
            return False
        try:
            return int(nums[0]) == int(expected)
        except (ValueError, TypeError):
            return False



EC_GAME_RULES = """You are given a graph. Your task is to count the number of edges in this graph.

An edge connects two nodes. Each edge is counted once (the graph is undirected).

Provide your answer as a single integer: the total number of edges.

Think step by step. Look at each node's neighbor list, count carefully, and remember that each edge appears in two neighbor lists.
"""


def _build_ec_prompt(G):
    g_text = _graph_to_text(G, "G")
    return f"{EC_GAME_RULES}\n\n{g_text}"



class EdgeCountingEnv(Env):
    def __init__(self):
        super().__init__(
            name="edge_counting",
            verifier=EdgeCountingVerifier,
        )

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> list:
        params = EC_DIFFICULTY_MAP.get(difficulty, EC_DIFFICULTY_MAP[2]).copy()
        n = params["n_nodes"]
        p = params["edge_prob"]

        rng = random.Random(seed)
        dataset = []

        for _ in range(num_of_questions):
            # Generate graph, ensure at least 1 edge
            for _ in range(max_attempts):
                G_raw = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 2**31))
                G = nx.relabel_nodes(G_raw, {i: i + 1 for i in range(n)})
                if G.number_of_edges() >= 1:
                    break

            edge_count = G.number_of_edges()
            question = _build_ec_prompt(G)

            dataset.append(Data(
                question=question,
                answer=str(edge_count),
                difficulty=difficulty,
                metadata={
                    "edge_count": edge_count,
                    "n_nodes": n,
                    "n_edges": edge_count,
                },
            ))

        return dataset



SYSTEM_PROMPT = (
    "Respond in the following format:\n"
    "<think>\n...\n</think>\n"
    "<answer>\n...\n</answer>"
)


def prepare_ec_dataset(difficulty=2, num_questions=500, seed=42, config=None):
    """Prepare HF Dataset for edge counting GRPO training."""
    if config is not None:
        difficulty = config.get("difficulty", difficulty)
        num_questions = config.get("num_questions", num_questions)
        seed = config.get("seed", seed)

    env = EdgeCountingEnv()
    data_list = env.generate(
        num_of_questions=num_questions,
        difficulty=difficulty,
        seed=seed,
    )

    dataset = []
    for d in data_list:
        dataset.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": d.question},
            ],
            "edge_count": d.answer,  # string of integer
        })

    random.seed(seed)
    random.shuffle(dataset)

    from datasets import Dataset
    return Dataset.from_list(dataset)


if __name__ == "__main__":
    env = EdgeCountingEnv()
    verifier = EdgeCountingVerifier()

    data_list = env.generate(num_of_questions=5, difficulty=2, seed=42)
    print(f"Generated {len(data_list)} edge counting problems (difficulty 2)")
    for i, d in enumerate(data_list):
        print(f"\n--- Problem {i+1} ---")
        print(f"Edge count: {d.metadata['edge_count']}")
        print(d.question[:200])

        # Verify correct answer
        sol = f"<think>Counting edges</think><answer>{d.answer}</answer>"
        assert verifier.verify(d, sol), f"Correct answer {d.answer} rejected!"

        # Verify wrong answer
        wrong = f"<think>Guessing</think><answer>{int(d.answer) + 1}</answer>"
        assert not verifier.verify(d, wrong), "Wrong answer accepted!"

    print("\n=== All edge counting sanity checks passed ===")
