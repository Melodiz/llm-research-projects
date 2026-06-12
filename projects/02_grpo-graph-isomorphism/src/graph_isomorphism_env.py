"""
Graph Isomorphism Environment for GRPO Training (HW2)
"""

import re
import json
import random
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np




@dataclass
class Data:
    question: str
    answer: str
    difficulty: int = 1
    metadata: dict = field(default_factory=dict)
    gpt_response: Optional[str] = None

    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }


class Verifier(ABC):
    @abstractmethod
    def verify(self, data: Data, test_answer: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, test_solution: str) -> str:
        raise NotImplementedError


class Env(ABC):
    def __init__(self, name: str, verifier: Verifier):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(self, num_of_questions=100, max_attempts=100, difficulty=1):
        raise NotImplementedError

    def verify(self, data: Data, test_solution: str) -> bool:
        return self.verifier.verify(data, test_solution)

    def extract_answer(self, test_solution: str) -> str:
        return self.verifier.extract_answer(test_solution)




class GraphIsomorphismVerifier(Verifier):
    """
    Bulletproof verifier for graph isomorphism.

    Handles:
    - Isomorphic pairs: validate ANY correct permutation (not just ground truth)
    - Non-isomorphic pairs: accept "NOT ISOMORPHIC" declaration
    - Automorphisms: multiple valid mappings accepted
    - LLM format deviations: 0-indexed, extra text, partial output
    """

    # --- 1a. Extract <answer> block ---

    ANSWER_PATTERN = re.compile(
        r"<answer>\s*(.*?)\s*</answer>",
        re.DOTALL | re.IGNORECASE,
    )
    ANSWER_UNCLOSED = re.compile(
        r"<answer>\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract answer from LLM generation containing <think>...<answer>... tags.
        Returns raw string from <answer> block.
        """
        if not test_solution or not isinstance(test_solution, str):
            return ""

        # Try all complete <answer>...</answer> blocks, take LAST
        matches = list(self.ANSWER_PATTERN.finditer(test_solution))
        if matches:
            return matches[-1].group(1).strip()

        # Fallback: unclosed <answer> tag (LLM ran out of tokens)
        unclosed = list(self.ANSWER_UNCLOSED.finditer(test_solution))
        if unclosed:
            return unclosed[-1].group(1).strip()

        return ""

    # --- 1b. Parse mapping from answer text ---

    @staticmethod
    def _parse_mapping(answer_text: str, n: int) -> Optional[dict]:
        """Parse vertex mapping from answer text."""
        if not answer_text:
            return None

        text = answer_text.strip()

        # Strategy 1: JSON dict {"1": "2", ...} or {1: 2, ...}
        mapping = GraphIsomorphismVerifier._try_json_dict(text, n)
        if mapping is not None:
            return mapping

        # Strategy 2: Permutation as list [2, 4, 6, 1, 3, 5]
        mapping = GraphIsomorphismVerifier._try_permutation_list(text, n)
        if mapping is not None:
            return mapping

        # Strategy 3: Arrow format "1->2, 3->4" or "1→2, 3→4"
        mapping = GraphIsomorphismVerifier._try_arrow_format(text, n)
        if mapping is not None:
            return mapping

        # Strategy 4: Colon format "1:2 3:4" or "1: 2, 3: 4"
        mapping = GraphIsomorphismVerifier._try_colon_format(text, n)
        if mapping is not None:
            return mapping

        # Strategy 5: Brute-force: find all (int, int) pairs in text
        mapping = GraphIsomorphismVerifier._try_pair_extraction(text, n)
        if mapping is not None:
            return mapping

        return None

    @staticmethod
    def _try_json_dict(text: str, n: int) -> Optional[dict]:
        """Parse JSON-like dict. Handles both string and int keys."""
        # Find JSON-like content between { }
        match = re.search(r'\{([^}]+)\}', text)
        if not match:
            return None
        try:
            # Fix common LLM issues: single quotes, trailing commas
            json_str = '{' + match.group(1) + '}'
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*}', '}', json_str)
            d = json.loads(json_str)
            mapping = {int(k): int(v) for k, v in d.items()}
            return GraphIsomorphismVerifier._validate_and_reindex(mapping, n)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    @staticmethod
    def _try_permutation_list(text: str, n: int) -> Optional[dict]:
        """Parse permutation as list: [2, 4, 6, 1, 3, 5] meaning node i -> perm[i]."""
        match = re.search(r'\[([^\]]+)\]', text)
        if not match:
            return None
        try:
            values = [int(x.strip()) for x in match.group(1).split(',')]
            if len(values) != n:
                return None
            # Heuristic: if 0 appears in values and n doesn't, it's 0-indexed
            if 0 in values and n not in values:
                # 0-indexed: node i -> values[i], convert to 1-indexed
                mapping = {i + 1: values[i] + 1 for i in range(n)}
            else:
                # 1-indexed: node i -> values[i-1]
                mapping = {i + 1: values[i] for i in range(n)}
            return GraphIsomorphismVerifier._validate_and_reindex(mapping, n)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _try_arrow_format(text: str, n: int) -> Optional[dict]:
        """Parse '1->2, 3->4' or '1→2' or '1 -> 2' format."""
        pairs = re.findall(r'(\d+)\s*(?:->|→|=>|maps?\s*to)\s*(\d+)', text, re.IGNORECASE)
        if len(pairs) < 1:
            return None
        try:
            mapping = {int(a): int(b) for a, b in pairs}
            return GraphIsomorphismVerifier._validate_and_reindex(mapping, n)
        except ValueError:
            return None

    @staticmethod
    def _try_colon_format(text: str, n: int) -> Optional[dict]:
        """Parse '1:2, 3:4' or '1: 2  3: 4' format."""
        pairs = re.findall(r'(\d+)\s*:\s*(\d+)', text)
        if len(pairs) < 1:
            return None
        try:
            mapping = {int(a): int(b) for a, b in pairs}
            return GraphIsomorphismVerifier._validate_and_reindex(mapping, n)
        except ValueError:
            return None

    @staticmethod
    def _try_pair_extraction(text: str, n: int) -> Optional[dict]:
        """
        Last resort: find all number pairs and try to build a mapping.
        """
        numbers = [int(x) for x in re.findall(r'\d+', text)]
        if len(numbers) < 2 * n:
            return None
        # Take first 2n numbers as n pairs
        try:
            mapping = {}
            for i in range(0, 2 * n, 2):
                mapping[numbers[i]] = numbers[i + 1]
            return GraphIsomorphismVerifier._validate_and_reindex(mapping, n)
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _validate_and_reindex(mapping: dict, n: int) -> Optional[dict]:
        """
        Validate mapping and handle 0-indexed → 1-indexed conversion.

        Checks:
        1. Correct number of entries (== n)
        2. No duplicate targets (bijection)
        3. All nodes in valid range

        Returns 1-indexed mapping or None.
        """
        if len(mapping) != n:
            return None

        keys = set(mapping.keys())
        vals = set(mapping.values())

        # Check for 0-indexed (keys are {0,...,n-1})
        if keys == set(range(n)) and vals.issubset(set(range(n))):
            # Convert to 1-indexed
            mapping = {k + 1: v + 1 for k, v in mapping.items()}
            keys = set(mapping.keys())
            vals = set(mapping.values())

        # Validate 1-indexed mapping
        expected = set(range(1, n + 1))
        if keys != expected:
            return None
        if vals != expected:
            return None  # duplicate targets or out-of-range
        if len(vals) != n:
            return None  # redundant but explicit: bijection check

        return mapping

    # --- 1c. Detect "NOT ISOMORPHIC" declaration ---

    NOT_ISO_PATTERNS = [
        re.compile(r'not\s+isomorphic', re.IGNORECASE),
        re.compile(r'non[\-\s]?isomorphic', re.IGNORECASE),
        re.compile(r'no\s+isomorphism', re.IGNORECASE),
        re.compile(r'graphs?\s+are\s+not\s+isomorphic', re.IGNORECASE),
        re.compile(r'NOT_ISOMORPHIC', re.IGNORECASE),
        re.compile(r'false', re.IGNORECASE),  # some LLMs just say "False"
    ]

    @classmethod
    def _is_not_isomorphic_declaration(cls, answer_text: str) -> bool:
        """Check if the answer declares non-isomorphism."""
        for pattern in cls.NOT_ISO_PATTERNS:
            if pattern.search(answer_text):
                return True
        return False



    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Core verification logic. Handle isomorphic and non-isomorphic cases.
        """
        answer_text = self.extract_answer(test_solution)
        if not answer_text:
            return False

        meta = data.metadata or {}
        is_isomorphic = meta.get("is_isomorphic", True)
        n = meta.get("n_nodes", 0)
        # Edge sets stored as sorted list of tuples for O(E) comparison
        g1_edges = set(tuple(e) for e in meta.get("g1_edges", []))
        g2_edges = set(tuple(e) for e in meta.get("g2_edges", []))

        claims_not_iso = self._is_not_isomorphic_declaration(answer_text)

        # Case 1: Ground truth = non-isomorphic
        if not is_isomorphic:
            if claims_not_iso:
                return True  # Correct!
            # LLM provided a mapping for non-iso pair → must be wrong
            # We could check the mapping, but that's wasteful. Just return False.
            return False

        # Case 2: Ground truth = isomorphic
        if claims_not_iso:
            return False  # Wrong — graphs ARE isomorphic

        # Case 3: Isomorphic + mapping provided → verify mapping
        mapping = self._parse_mapping(answer_text, n)
        if mapping is None:
            return False  # Couldn't parse a valid mapping

        return self._verify_mapping(mapping, g1_edges, g2_edges, n)

    @staticmethod
    def _verify_mapping(
        mapping: dict,
        g1_edges: set,
        g2_edges: set,
        n: int,
    ) -> bool:
        """
        Apply mapping to G1's edges and check equality with G2's edges.
        Time: O(|E|). No need for O(V!) brute force.

        mapping: {g1_node → g2_node}
        g1_edges: set of (u, v) tuples (1-indexed, u < v)
        g2_edges: set of (u, v) tuples (1-indexed, u < v)
        """
        # Apply mapping: for each edge (u,v) in G1, the mapped edge
        # should be (mapping[u], mapping[v]) and must exist in G2
        mapped_edges = set()
        for u, v in g1_edges:
            mu, mv = mapping.get(u), mapping.get(v)
            if mu is None or mv is None:
                return False
            # Normalize edge direction (store as min, max)
            mapped_edges.add((min(mu, mv), max(mu, mv)))

        return mapped_edges == g2_edges

    # --- 2b. Partial credit computation (for reward variants) ---

    @staticmethod
    def _partial_credit(
        mapping: dict,
        g1_edges: set,
        g2_edges: set,
        n: int,
    ) -> float:
        """
        Fraction of G2 edges correctly reproduced by the proposed mapping.
        Returns float in [0.0, 1.0].
        """
        if not g2_edges:
            return 1.0  # Edge-free graph: any bijection works
        if not mapping or len(mapping) != n:
            return 0.0

        mapped_edges = set()
        for u, v in g1_edges:
            mu, mv = mapping.get(u), mapping.get(v)
            if mu is not None and mv is not None:
                mapped_edges.add((min(mu, mv), max(mu, mv)))

        correct = len(mapped_edges & g2_edges)
        return correct / len(g2_edges)




def binary_reward(completions, ground_truth, metadata_list, **kwargs):
    """
    Standard binary reward. 1.0 if correct, 0.0 otherwise.
    """
    verifier = GraphIsomorphismVerifier()
    rewards = []
    for completion, gt, meta in zip(completions, ground_truth, metadata_list):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        data = Data(question="", answer=gt, metadata=meta)
        rewards.append(1.0 if verifier.verify(data, text) else 0.0)
    return rewards


def partial_credit_reward(completions, ground_truth, metadata_list, **kwargs):
    """
    Partial credit for isomorphic pairs: fraction of edges matched.
    Non-isomorphic pairs: binary (1.0 correct / 0.0 wrong).
    """
    verifier = GraphIsomorphismVerifier()
    rewards = []
    for completion, gt, meta in zip(completions, ground_truth, metadata_list):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        data = Data(question="", answer=gt, metadata=meta)

        is_iso = meta.get("is_isomorphic", True)
        n = meta.get("n_nodes", 0)
        g1_edges = set(tuple(e) for e in meta.get("g1_edges", []))
        g2_edges = set(tuple(e) for e in meta.get("g2_edges", []))

        answer_text = verifier.extract_answer(text)

        if not is_iso:
            # Non-isomorphic: binary only
            claims_not = verifier._is_not_isomorphic_declaration(answer_text)
            rewards.append(1.0 if claims_not else 0.0)
        else:
            # Isomorphic: check if fully correct first
            if verifier.verify(data, text):
                rewards.append(1.0)
            else:
                # Partial credit
                mapping = verifier._parse_mapping(answer_text, n)
                if mapping is None:
                    rewards.append(0.0)
                else:
                    pc = verifier._partial_credit(mapping, g1_edges, g2_edges, n)
                    rewards.append(pc * 0.8)
    return rewards


def format_reward(completions, **kwargs):
    """
    Small bonus for valid format: <think>...</think><answer>...</answer>
    """
    pattern = re.compile(
        r"<think>.*?</think>\s*<answer>.*?</answer>",
        re.DOTALL | re.IGNORECASE,
    )
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(0.1 if pattern.search(text) else 0.0)
    return rewards


def composite_reward(completions, ground_truth, metadata_list, **kwargs):
    """
    Recommended reward: binary correctness + format bonus.

    This avoids the partial credit pitfalls while still providing
    gradient signal via format reward.
    """
    verifier = GraphIsomorphismVerifier()
    fmt_pattern = re.compile(
        r"<think>.*?</think>\s*<answer>.*?</answer>",
        re.DOTALL | re.IGNORECASE,
    )
    rewards = []
    for completion, gt, meta in zip(completions, ground_truth, metadata_list):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        data = Data(question="", answer=gt, metadata=meta)

        r = 0.0
        # Format bonus
        if fmt_pattern.search(text):
            r += 0.1
        # Correctness
        if verifier.verify(data, text):
            r += 1.0
        rewards.append(r)
    return rewards




# --- 4a. Difficulty → Hyperparameter Mapping ---
# Difficulty 1-10 maps to (n_nodes, edge_prob, iso_ratio, non_iso_hardness)

DIFFICULTY_MAP = {
    # diff: (n_nodes, edge_prob, iso_ratio, non_iso_category)
    #
    # iso_ratio: fraction of isomorphic pairs (0.5 = balanced)
    # non_iso_category: 'easy' | 'medium' | 'hard'
    #
    # n_nodes is the PRIMARY difficulty knob.
    # edge_prob secondary (denser = harder for LLM spatial reasoning).
    #
    1:  {"n_nodes": 3,  "edge_prob": 0.4, "iso_ratio": 0.5, "non_iso": "easy"},
    2:  {"n_nodes": 4,  "edge_prob": 0.4, "iso_ratio": 0.5, "non_iso": "easy"},
    3:  {"n_nodes": 4,  "edge_prob": 0.5, "iso_ratio": 0.5, "non_iso": "easy"},
    4:  {"n_nodes": 5,  "edge_prob": 0.4, "iso_ratio": 0.5, "non_iso": "medium"},
    5:  {"n_nodes": 5,  "edge_prob": 0.5, "iso_ratio": 0.5, "non_iso": "medium"},
    6:  {"n_nodes": 6,  "edge_prob": 0.4, "iso_ratio": 0.5, "non_iso": "medium"},
    7:  {"n_nodes": 6,  "edge_prob": 0.5, "iso_ratio": 0.5, "non_iso": "hard"},
    8:  {"n_nodes": 7,  "edge_prob": 0.5, "iso_ratio": 0.5, "non_iso": "hard"},
    9:  {"n_nodes": 8,  "edge_prob": 0.5, "iso_ratio": 0.5, "non_iso": "hard"},
    10: {"n_nodes": 9,  "edge_prob": 0.5, "iso_ratio": 0.5, "non_iso": "hard"},
}


def _edges_to_sorted_tuples(G):
    """Convert nx graph edges to sorted list of (min, max) tuples."""
    return sorted((min(u, v), max(u, v)) for u, v in G.edges())


def _graph_to_text(G, name="G"):
    """
    Convert graph to text representation for LLM prompt.

    Format: adjacency list, 1-indexed.
    Example:
        Graph G1 (5 nodes):
        Node 1: neighbors [2, 3]
        Node 2: neighbors [1, 4, 5]
        ...
    """
    n = G.number_of_nodes()
    lines = [f"Graph {name} ({n} nodes, {G.number_of_edges()} edges):"]
    for node in sorted(G.nodes()):
        neighbors = sorted(G.neighbors(node))
        if neighbors:
            lines.append(f"  Node {node}: neighbors {neighbors}")
        else:
            lines.append(f"  Node {node}: no neighbors")
    return "\n".join(lines)


# --- 4b. Isomorphic pair generation ---

def _generate_isomorphic_pair(n, edge_prob, rng):
    """
    Generate (G1, G2, permutation) where G2 = permute(G1).

    Returns: (G1, G2, perm_dict, perm_str)
        G1, G2: nx.Graph with nodes 1..n
        perm_dict: {g1_node: g2_node}
        perm_str: string representation for Data.answer
    """
    # Generate G1 as Erdos-Renyi
    # We relabel to 1-indexed immediately.
    G1_raw = nx.erdos_renyi_graph(n, edge_prob, seed=rng.randint(0, 2**31))
    G1 = nx.relabel_nodes(G1_raw, {i: i + 1 for i in range(n)})

    # Random permutation for G2
    nodes = list(range(1, n + 1))
    shuffled = nodes.copy()
    rng.shuffle(shuffled)
    perm = dict(zip(nodes, shuffled))  # perm[g1_node] = g2_node

    # Apply permutation to create G2
    G2 = nx.relabel_nodes(G1, perm)

    # Format permutation as answer string
    perm_str = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))

    return G1, G2, perm, perm_str


# --- 4c. Non-isomorphic pair generation ---

def _generate_non_isomorphic_easy(n, edge_prob, rng):
    """
    Easy non-iso: different number of edges (same node count).
    """
    G1_raw = nx.erdos_renyi_graph(n, edge_prob, seed=rng.randint(0, 2**31))
    G1 = nx.relabel_nodes(G1_raw, {i: i + 1 for i in range(n)})

    # G2: same n but different edge_prob → likely different edge count
    alt_prob = edge_prob + rng.uniform(0.15, 0.35)
    alt_prob = min(alt_prob, 0.95)
    for _ in range(100):
        G2_raw = nx.erdos_renyi_graph(n, alt_prob, seed=rng.randint(0, 2**31))
        G2 = nx.relabel_nodes(G2_raw, {i: i + 1 for i in range(n)})
        if G1.number_of_edges() != G2.number_of_edges():
            if not nx.is_isomorphic(G1, G2):
                return G1, G2
    # Fallback: different node count (trivially non-iso)
    G2_raw = nx.erdos_renyi_graph(n + 1, edge_prob, seed=rng.randint(0, 2**31))
    G2 = nx.relabel_nodes(G2_raw, {i: i + 1 for i in range(n + 1)})
    return G1, G2


def _generate_non_isomorphic_medium(n, edge_prob, rng):
    """
    Medium non-iso: same degree sequence, different structure.
    """
    G1_raw = nx.erdos_renyi_graph(n, edge_prob, seed=rng.randint(0, 2**31))
    G1 = nx.relabel_nodes(G1_raw, {i: i + 1 for i in range(n)})
    deg_seq = sorted([d for _, d in G1.degree()], reverse=True)

    for _ in range(100):
        try:
            # Configuration model → simple graph
            G2_raw = nx.random_degree_sequence_graph(deg_seq, seed=rng.randint(0, 2**31))
            if G2_raw.number_of_nodes() != n:
                continue
            G2 = nx.relabel_nodes(G2_raw, {i: i + 1 for i in range(n)})
            if not nx.is_isomorphic(G1, G2):
                return G1, G2
        except nx.NetworkXUnfeasible:
            continue
        except nx.NetworkXError:
            continue

    # Fallback to easy
    return _generate_non_isomorphic_easy(n, edge_prob, rng)


def _generate_non_isomorphic_hard(n, edge_prob, rng):
    """
    Hard non-iso: strongly regular graphs or cospectral mates.
    """
    # Try to find strongly regular graph pairs
    if n >= 10:
        # Petersen graph (srg(10,3,0,1))
        G1 = nx.relabel_nodes(nx.petersen_graph(), {i: i + 1 for i in range(10)})
        # Create a non-isomorphic graph with same parameters
        # The complement of Petersen is the Kneser graph K(5,2) which IS
        # isomorphic to Petersen. Instead, use a different srg if available.
        # Fallback: medium-difficulty approach
        return _generate_non_isomorphic_medium(n, edge_prob, rng)

    if n >= 6:
        # For n=6: Frucht's approach — find graphs with identical
        # degree sequences but different local structure
        # Practical approach: generate many random graphs with same
        # degree sequence and find non-isomorphic pair
        return _generate_non_isomorphic_medium(n, edge_prob, rng)

    return _generate_non_isomorphic_medium(n, edge_prob, rng)


# --- 4d. Prompt template ---

GAME_RULES = """You are given two graphs G1 and G2. Your task is to determine whether they are isomorphic.

Two graphs are isomorphic if there exists a one-to-one mapping (bijection) between their vertex sets that preserves edges. That is, if vertex u is connected to vertex v in G1, then the mapped vertices must also be connected in G2, and vice versa.

If the graphs ARE isomorphic:
- Provide the vertex mapping as a list of arrows: "1->X, 2->Y, 3->Z, ..."
  where each entry means "node i in G1 maps to node X in G2"
- The mapping must be a valid permutation (each G2 node appears exactly once)
- Nodes are 1-indexed

If the graphs are NOT isomorphic:
- Write exactly: NOT ISOMORPHIC

Think step by step. Consider the number of nodes, edges, and degree sequences first. If those match, try to construct a mapping.
"""


def _build_prompt(G1, G2):
    """Build the full question string for Data."""
    g1_text = _graph_to_text(G1, "G1")
    g2_text = _graph_to_text(G2, "G2")
    return f"{GAME_RULES}\n\n{g1_text}\n\n{g2_text}"

def _build_prompt_with_hints(G1, G2, perm, k, rng=None):
    """Build the prompt with k mapping hints revealed."""
    base_prompt = _build_prompt(G1, G2)
    if k <= 0 or not perm:
        return base_prompt
        
    keys = list(perm.keys())
    if rng:
        sample_keys = rng.sample(keys, min(k, len(keys)))
    else:
        sample_keys = random.sample(keys, min(k, len(keys)))
        
    sample_keys.sort() # For consistent reading order
    
    hints = []
    for u in sample_keys:
        v = perm[u]
        hints.append(f"Node {u} in G1 maps to Node {v} in G2")
        
    hint_text = "Hint: " + ", ".join(hints) + "."
    return f"{base_prompt}\n\n{hint_text}"




class GraphIsomorphismEnv(Env):
    def __init__(self):
        super().__init__(
            name="graph_isomorphism",
            verifier=GraphIsomorphismVerifier,
        )

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: int = 5,
        # Direct hyperparameter override (HW2 requirement)
        n_nodes: int = None,
        edge_prob: float = None,
        iso_ratio: float = None,
        non_iso: str = None,
        seed: int = 42,
        hints_k: int = None,
    ) -> list:
        """
        Generate graph isomorphism instances.

        Supports both:
        1. difficulty (int 1-10) → maps to hyperparameters via DIFFICULTY_MAP
        2. Direct hyperparameters (n_nodes, edge_prob, iso_ratio, non_iso)
        """
        # Resolve hyperparameters
        params = DIFFICULTY_MAP.get(difficulty, DIFFICULTY_MAP[5]).copy()
        if n_nodes is not None:
            params["n_nodes"] = n_nodes
        if edge_prob is not None:
            params["edge_prob"] = edge_prob
        if iso_ratio is not None:
            params["iso_ratio"] = iso_ratio
        if non_iso is not None:
            params["non_iso"] = non_iso

        n = params["n_nodes"]
        p = params["edge_prob"]
        ratio = params["iso_ratio"]
        non_iso_cat = params["non_iso"]

        rng = random.Random(seed)
        dataset = []

        num_iso = int(num_of_questions * ratio)
        num_non_iso = num_of_questions - num_iso

        # Generate isomorphic pairs
        for _ in range(num_iso):
            for attempt in range(max_attempts):
                G1, G2, perm, perm_str = _generate_isomorphic_pair(n, p, rng)
                # Reject trivially empty graphs
                if G1.number_of_edges() >= 1:
                    break
            else:
                G1, G2, perm, perm_str = _generate_isomorphic_pair(n, p, rng)

            if hints_k is not None and hints_k > 0:
                question = _build_prompt_with_hints(G1, G2, perm, hints_k, rng)
            else:
                question = _build_prompt(G1, G2)
                
            answer = perm_str  # Ground truth mapping (but verifier accepts any valid one)
            metadata = {
                "is_isomorphic": True,
                "n_nodes": G1.number_of_nodes(),
                "g1_edges": _edges_to_sorted_tuples(G1),
                "g2_edges": _edges_to_sorted_tuples(G2),
                "ground_truth_perm": perm,
            }
            dataset.append(Data(
                question=question,
                answer=answer,
                difficulty=difficulty,
                metadata=metadata,
            ))

        # Generate non-isomorphic pairs
        generators = {
            "easy": _generate_non_isomorphic_easy,
            "medium": _generate_non_isomorphic_medium,
            "hard": _generate_non_isomorphic_hard,
        }
        gen_func = generators.get(non_iso_cat, _generate_non_isomorphic_medium)

        for _ in range(num_non_iso):
            for attempt in range(max_attempts):
                result = gen_func(n, p, rng)
                G1, G2 = result
                if G1.number_of_edges() >= 1 and G2.number_of_edges() >= 1:
                    # Double-check non-isomorphism
                    if not nx.is_isomorphic(G1, G2):
                        break
            else:
                # Last resort: different number of nodes
                G1_raw = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 2**31))
                G1 = nx.relabel_nodes(G1_raw, {i: i + 1 for i in range(n)})
                G2_raw = nx.erdos_renyi_graph(n + 1, p, seed=rng.randint(0, 2**31))
                G2 = nx.relabel_nodes(G2_raw, {i: i + 1 for i in range(n + 1)})

            question = _build_prompt(G1, G2)
            answer = "NOT ISOMORPHIC"
            metadata = {
                "is_isomorphic": False,
                "n_nodes": max(G1.number_of_nodes(), G2.number_of_nodes()),
                "g1_edges": _edges_to_sorted_tuples(G1),
                "g2_edges": _edges_to_sorted_tuples(G2),
                "g1_n": G1.number_of_nodes(),
                "g2_n": G2.number_of_nodes(),
            }
            dataset.append(Data(
                question=question,
                answer=answer,
                difficulty=difficulty,
                metadata=metadata,
            ))

        # Shuffle to mix iso/non-iso
        rng.shuffle(dataset)
        return dataset

    def extract_answer(self, test_solution: str) -> str:
        return self.verifier.extract_answer(test_solution)




def make_correctness_reward_func(reward_type: str = "binary"):
    """
    Factory for trl-compatible reward functions.

    Usage with GRPOTrainer:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                make_correctness_reward_func("binary"),
                format_reward,
            ],
            args=GRPOConfig(..., reward_weights=[1.0, 0.1]),
            train_dataset=dataset,
        )

    """
    reward_fn = {
        "binary": binary_reward,
        "partial": partial_credit_reward,
        "composite": composite_reward,
    }[reward_type]

    def wrapped(completions, **kwargs):
        return reward_fn(completions, **kwargs)
    return wrapped




def prepare_grpo_dataset(
    env: GraphIsomorphismEnv,
    num_questions: int = 500,
    difficulty: int = 5,
    seed: int = 42,
    iso_ratio: float = None,
) -> list:
    """Generate dataset in format compatible with trl GRPOTrainer."""
    SYSTEM_PROMPT = (
        "Respond in the following format:\n"
        "<think>\n...\n</think>\n"
        "<answer>\n...\n</answer>"
    )

    gen_kwargs = dict(
        num_of_questions=num_questions,
        difficulty=difficulty,
        seed=seed,
    )
    if iso_ratio is not None:
        gen_kwargs["iso_ratio"] = iso_ratio

    data_list = env.generate(**gen_kwargs)

    dataset = []
    for d in data_list:
        is_iso = d.metadata.get("is_isomorphic", True)
        dataset.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": d.question},
            ],
            "ground_truth": d.answer,
            "metadata_list": d.metadata,
            "task_type": "isomorphic" if is_iso else "non_isomorphic",
        })

    return dataset







def _run_sanity_checks():
    """
    Sanity checks for the environment. Run after any code changes.

    Tests:
    1. Isomorphic pair → correct mapping → reward 1.0
    2. Isomorphic pair → wrong mapping → reward 0.0
    3. Isomorphic pair → "NOT ISOMORPHIC" → reward 0.0
    4. Non-isomorphic pair → "NOT ISOMORPHIC" → reward 1.0
    5. Non-isomorphic pair → any mapping → reward 0.0
    6. Automorphism: complete graph K3, multiple valid mappings
    7. Parsing: 0-indexed input → correct handling
    8. Parsing: JSON dict format
    9. Parsing: arrow format
    10. Empty/malformed input → reward 0.0
    """
    env = GraphIsomorphismEnv()
    verifier = GraphIsomorphismVerifier()

    # --- Test 1: Correct mapping ---
    data_list = env.generate(num_of_questions=5, difficulty=3, seed=42)
    iso_data = [d for d in data_list if d.metadata["is_isomorphic"]]
    if iso_data:
        d = iso_data[0]
        perm = d.metadata["ground_truth_perm"]
        perm_str = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))
        solution = f"<think>Checking edges...</think><answer>{perm_str}</answer>"
        assert verifier.verify(d, solution), "Test 1 FAILED: correct mapping rejected"
        print("✓ Test 1: correct mapping accepted")

    # --- Test 2: Wrong mapping ---
    if iso_data:
        d = iso_data[0]
        n = d.metadata["n_nodes"]
        # Reverse the permutation (wrong unless graph is self-complementary)
        wrong_perm = {i: n + 1 - i for i in range(1, n + 1)}
        wrong_str = ", ".join(f"{k}->{v}" for k, v in sorted(wrong_perm.items()))
        solution = f"<think>Guessing...</think><answer>{wrong_str}</answer>"
        result = verifier.verify(d, solution)
        # Note: might actually be correct by chance for small graphs!
        print(f"✓ Test 2: wrong mapping → {'accepted (automorphism!)' if result else 'rejected'}")

    # --- Test 3: "NOT ISOMORPHIC" for isomorphic pair ---
    if iso_data:
        d = iso_data[0]
        solution = "<think>Hmm</think><answer>NOT ISOMORPHIC</answer>"
        assert not verifier.verify(d, solution), "Test 3 FAILED"
        print("✓ Test 3: NOT ISOMORPHIC for iso pair → rejected")

    # --- Test 4: Correct non-iso ---
    non_iso_data = [d for d in data_list if not d.metadata["is_isomorphic"]]
    if non_iso_data:
        d = non_iso_data[0]
        solution = "<think>Different degree sequences</think><answer>NOT ISOMORPHIC</answer>"
        assert verifier.verify(d, solution), "Test 4 FAILED"
        print("✓ Test 4: NOT ISOMORPHIC for non-iso pair → accepted")

    # --- Test 5: Mapping for non-iso pair ---
    if non_iso_data:
        d = non_iso_data[0]
        n = d.metadata.get("g1_n", d.metadata["n_nodes"])
        fake = ", ".join(f"{i}->{i}" for i in range(1, n + 1))
        solution = f"<think>Trying</think><answer>{fake}</answer>"
        assert not verifier.verify(d, solution), "Test 5 FAILED"
        print("✓ Test 5: mapping for non-iso pair → rejected")

    # --- Test 6: Automorphism (K3) ---
    G1 = nx.complete_graph(3)
    G1 = nx.relabel_nodes(G1, {i: i + 1 for i in range(3)})
    G2 = G1.copy()
    edges1 = _edges_to_sorted_tuples(G1)
    edges2 = _edges_to_sorted_tuples(G2)
    meta = {"is_isomorphic": True, "n_nodes": 3, "g1_edges": edges1, "g2_edges": edges2}
    d = Data(question="", answer="1->1, 2->2, 3->3", metadata=meta)
    # All 6 permutations of {1,2,3} should be valid for K3
    import itertools as it
    valid_count = 0
    for perm in it.permutations([1, 2, 3]):
        perm_str = ", ".join(f"{i+1}->{p}" for i, p in enumerate(perm))
        sol = f"<think>K3</think><answer>{perm_str}</answer>"
        if verifier.verify(d, sol):
            valid_count += 1
    assert valid_count == 6, f"Test 6 FAILED: only {valid_count}/6 K3 permutations accepted"
    print(f"✓ Test 6: all 6 automorphisms of K3 accepted")

    # --- Test 7: 0-indexed parsing ---
    meta = {"is_isomorphic": True, "n_nodes": 3,
            "g1_edges": [(1, 2), (2, 3)], "g2_edges": [(1, 3), (2, 3)]}
    d = Data(question="", answer="", metadata=meta)
    # 0-indexed: 0->2, 1->1, 2->0 means 1->3, 2->2, 3->1
    sol = "<think>0-indexed</think><answer>{0: 2, 1: 1, 2: 0}</answer>"
    mapping = verifier._parse_mapping("{0: 2, 1: 1, 2: 0}", 3)
    assert mapping == {1: 3, 2: 2, 3: 1}, f"Test 7 FAILED: got {mapping}"
    print("✓ Test 7: 0-indexed mapping correctly converted to 1-indexed")

    # --- Test 8: JSON format ---
    mapping = verifier._parse_mapping('{"1": "3", "2": "1", "3": "2"}', 3)
    assert mapping == {1: 3, 2: 1, 3: 2}, f"Test 8 FAILED: got {mapping}"
    print("✓ Test 8: JSON dict format parsed correctly")

    # --- Test 9: Arrow format ---
    mapping = verifier._parse_mapping("1 -> 3, 2 -> 1, 3 -> 2", 3)
    assert mapping == {1: 3, 2: 1, 3: 2}, f"Test 9 FAILED: got {mapping}"
    print("✓ Test 9: arrow format parsed correctly")

    # --- Test 10: Malformed input ---
    assert verifier.extract_answer("") == ""
    assert verifier.extract_answer("no tags here") == ""
    assert not verifier.verify(
        Data(question="", answer="", metadata={"is_isomorphic": True, "n_nodes": 3,
             "g1_edges": [], "g2_edges": []}),
        "garbage"
    )
    print("✓ Test 10: malformed input handled gracefully")

    print("\n=== All sanity checks passed ===")


if __name__ == "__main__":
    _run_sanity_checks()

    # Demo: generate a dataset and show one instance
    print("\n--- Demo instance ---")
    env = GraphIsomorphismEnv()
    dataset = env.generate(num_of_questions=3, difficulty=5, seed=123)
    d = dataset[0]
    print(f"Difficulty: {d.difficulty}")
    print(f"Is isomorphic: {d.metadata['is_isomorphic']}")
    print(f"Question (first 300 chars):\n{d.question[:300]}...")
    print(f"Answer: {d.answer}")
