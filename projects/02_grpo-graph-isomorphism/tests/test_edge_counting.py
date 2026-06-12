"""Tests for edge counting positive control."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edge_counting_env import (
    EdgeCountingEnv, EdgeCountingVerifier, EC_DIFFICULTY_MAP,
    prepare_ec_dataset,
)
from edge_counting_reward import ec_reward
from graph_isomorphism_env import Data




def _wrap(ans):
    return f"<think>reasoning</think><answer>{ans}</answer>"

def _wrap_trl(ans):
    return [{"role": "assistant", "content": _wrap(ans)}]




class TestEdgeCountingEnv:
    def test_generates_correct_count(self):
        env = EdgeCountingEnv()
        data_list = env.generate(num_of_questions=20, difficulty=2, seed=42)
        assert len(data_list) == 20
        for d in data_list:
            assert d.metadata["edge_count"] == int(d.answer)
            assert d.metadata["edge_count"] >= 1

    def test_difficulties(self):
        env = EdgeCountingEnv()
        for diff in (1, 2, 3):
            data_list = env.generate(num_of_questions=5, difficulty=diff, seed=42)
            expected_n = EC_DIFFICULTY_MAP[diff]["n_nodes"]
            for d in data_list:
                assert d.metadata["n_nodes"] == expected_n
                assert d.difficulty == diff

    def test_prompt_contains_graph(self):
        env = EdgeCountingEnv()
        data_list = env.generate(num_of_questions=3, difficulty=2, seed=42)
        for d in data_list:
            assert "Graph G" in d.question
            assert "Node" in d.question
            assert "edges" in d.question.lower()




class TestEdgeCountingVerifier:
    def test_correct_answer_accepted(self):
        v = EdgeCountingVerifier()
        data = Data(question="", answer="5", metadata={"edge_count": 5})
        sol = "<think>counting</think><answer>5</answer>"
        assert v.verify(data, sol)

    def test_wrong_answer_rejected(self):
        v = EdgeCountingVerifier()
        data = Data(question="", answer="5", metadata={"edge_count": 5})
        sol = "<think>counting</think><answer>3</answer>"
        assert not v.verify(data, sol)

    def test_no_tags_rejected(self):
        v = EdgeCountingVerifier()
        data = Data(question="", answer="5", metadata={"edge_count": 5})
        assert not v.verify(data, "the answer is 5")

    def test_extracts_integer_from_text(self):
        v = EdgeCountingVerifier()
        data = Data(question="", answer="3", metadata={"edge_count": 3})
        sol = "<answer>The number of edges is 3.</answer>"
        assert v.verify(data, sol)

    def test_empty_input(self):
        v = EdgeCountingVerifier()
        data = Data(question="", answer="5", metadata={"edge_count": 5})
        assert not v.verify(data, "")
        assert not v.verify(data, None)

    def test_with_generated_data(self):
        """Verifier agrees with ground truth for generated problems."""
        env = EdgeCountingEnv()
        v = EdgeCountingVerifier()
        for d in env.generate(num_of_questions=30, difficulty=3, seed=77):
            sol = f"<answer>{d.answer}</answer>"
            assert v.verify(d, sol), f"Rejected correct answer {d.answer}"




class TestEdgeCountingReward:
    def test_correct_with_format(self):
        """Correct + format: 0.7 * 1.0 + 0.3 * 0.5 = 0.85"""
        rewards = ec_reward([_wrap("5")], ["5"])
        assert rewards[0] == pytest.approx(0.85)

    def test_correct_no_format(self):
        """Correct without tags: 0.7 * 1.0 + 0.3 * 0.0 = 0.7"""
        rewards = ec_reward(["<answer>5</answer>"], ["5"])
        assert rewards[0] == pytest.approx(0.7)

    def test_wrong_with_format(self):
        """Wrong + format: 0.7 * (-1.0) + 0.3 * 0.5 = -0.55"""
        rewards = ec_reward([_wrap("3")], ["5"])
        assert rewards[0] == pytest.approx(-0.55)

    def test_wrong_no_format(self):
        """Wrong without tags: 0.7 * (-1.0) + 0.3 * 0.0 = -0.7"""
        rewards = ec_reward(["<answer>3</answer>"], ["5"])
        assert rewards[0] == pytest.approx(-0.7)

    def test_trl_format(self):
        """Works with trl >=0.16 list-of-dicts format."""
        rewards = ec_reward([_wrap_trl("5")], ["5"])
        assert rewards[0] == pytest.approx(0.85)

    def test_batch(self):
        """Batch of mixed correct/wrong."""
        comps = [_wrap("5"), _wrap("3"), _wrap("5")]
        ecs = ["5", "5", "3"]
        rewards = ec_reward(comps, ecs)
        assert len(rewards) == 3
        assert rewards[0] == pytest.approx(0.85)   # correct
        assert rewards[1] == pytest.approx(-0.55)   # wrong
        assert rewards[2] == pytest.approx(-0.55)   # wrong




class TestECDatasetPrep:
    def test_prepare_ec_dataset(self):
        ds = prepare_ec_dataset(difficulty=2, num_questions=20, seed=42)
        assert len(ds) == 20
        assert "prompt" in ds.column_names
        assert "edge_count" in ds.column_names

    def test_dataset_from_config(self):
        config = {"difficulty": 1, "num_questions": 10}
        ds = prepare_ec_dataset(config=config)
        assert len(ds) == 10
        for row in ds:
            assert int(row["edge_count"]) >= 1




class TestECConfig:
    def test_config_exists(self):
        from train import CONFIGS
        assert "ec" in CONFIGS

    def test_config_values(self):
        from train import CONFIGS
        c = CONFIGS["ec"]
        assert c["run_name"] == "run_ec"
        assert c["task"] == "edge_counting"
        assert c["reward_type"] == "ec"
        assert c["max_steps"] == 500
        assert c["scale_rewards"] == "batch"
        assert c["loss_type"] == "grpo"
