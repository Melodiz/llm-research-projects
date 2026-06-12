"""Tests for the hint generation mechanism in GraphIsomorphismEnv."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from graph_isomorphism_env import GraphIsomorphismEnv


def test_hints_generation_k_equals_n():
    """k=n (all hints revealed). Model has trivially solvable problem."""
    env = GraphIsomorphismEnv()
    # n=5 nodes, all isomorphic pairs
    dataset = env.generate(
        num_of_questions=10, n_nodes=5, iso_ratio=1.0, hints_k=5, seed=42
    )
    
    for d in dataset:
        assert d.metadata["is_isomorphic"] is True
        assert "Hint: " in d.question
        # Check exactly 5 mappings are revealed
        hint_line = [line for line in d.question.split('\n') if line.startswith('Hint: ')][0]
        assert hint_line.count("maps to") == 5


def test_hints_generation_k_equals_zero():
    """k=0 (or None). No hints are revealed, fallback to default behavior."""
    env = GraphIsomorphismEnv()
    dataset = env.generate(
        num_of_questions=10, n_nodes=5, iso_ratio=1.0, hints_k=0, seed=42
    )
    
    for d in dataset:
        assert "Hint: " not in d.question

    dataset_none = env.generate(
        num_of_questions=10, n_nodes=5, iso_ratio=1.0, hints_k=None, seed=42
    )
    for d in dataset_none:
        assert "Hint: " not in d.question


def test_hints_generation_k_equals_n_minus_2():
    """k=n-2. Reveals all but 2 mappings, model must figure out 2 remaining."""
    env = GraphIsomorphismEnv()
    dataset = env.generate(
        num_of_questions=10, n_nodes=5, iso_ratio=1.0, hints_k=3, seed=42
    )
    
    for d in dataset:
        assert "Hint: " in d.question
        hint_line = [line for line in d.question.split('\n') if line.startswith('Hint: ')][0]
        assert hint_line.count("maps to") == 3


def test_hints_only_on_isomorphic_pairs():
    """Non-isomorphic pairs should NEVER get hints, as they're misleading."""
    env = GraphIsomorphismEnv()
    dataset = env.generate(
        num_of_questions=20, n_nodes=5, iso_ratio=0.5, hints_k=3, seed=42
    )
    
    for d in dataset:
        if d.metadata["is_isomorphic"]:
            assert "Hint: " in d.question
            hint_line = [line for line in d.question.split('\n') if line.startswith('Hint: ')][0]
            assert hint_line.count("maps to") == 3
        else:
            assert "Hint: " not in d.question
