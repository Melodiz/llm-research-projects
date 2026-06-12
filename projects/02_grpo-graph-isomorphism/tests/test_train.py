"""Tests for train.py (Config parsing and dataset prep)."""

import sys
import os
import pytest
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import CONFIGS, prepare_dataset


class TestConfigs:
    def test_run1_config(self):
        c = CONFIGS["run1"]
        assert c["run_name"] == "run1_antihack"
        assert c["reward_type"] == "class_aware"
        assert c["scale_rewards"] == "batch"
        assert c["loss_type"] == "dapo"
        assert c["beta"] == 0.04
        assert c["iso_ratio"] == 0.75

    def test_run2_config(self):
        c = CONFIGS["run2"]
        assert c["run_name"] == "run2_vanilla"
        assert c["reward_type"] == "composite"
        assert c["scale_rewards"] == "group"
        assert c["loss_type"] == "grpo"
        assert c["beta"] == 0.0
        assert c["iso_ratio"] == 0.5

    def test_run2b_config(self):
        c = CONFIGS["run2b"]
        assert c["run_name"] == "run2b_batch_norm_only"
        assert c["reward_type"] == "composite"
        assert c["scale_rewards"] == "batch"
        assert c["loss_type"] == "grpo"
        assert c["beta"] == 0.0
        assert c["iso_ratio"] == 0.5

    def test_smoke_config(self):
        c = CONFIGS["smoke"]
        assert c["run_name"] == "smoke_test"
        assert c["max_steps"] == 3


class TestDatasetPrep:
    def test_prepare_dataset_run1_75_25(self):
        """Iso_ratio=0.75 should give 75 iso and 25 non_iso out of 100."""
        config = {
            "num_questions": 100,
            "difficulty": 1,
            "iso_ratio": 0.75,
        }
        ds = prepare_dataset(config=config)
        assert isinstance(ds, Dataset)
        assert len(ds) == 100

        counts = {"isomorphic": 0, "non_isomorphic": 0}
        for item in ds:
            counts[item["task_type"]] += 1

        assert counts["isomorphic"] == 75
        assert counts["non_isomorphic"] == 25

        # Check standard columns
        for col in ["prompt", "ground_truth", "metadata_list", "task_type"]:
            assert col in ds.column_names

    def test_prepare_dataset_run2_50_50(self):
        """Iso_ratio=0.5 should give 50/50 split."""
        config = {
            "num_questions": 100,
            "difficulty": 1,
            "iso_ratio": 0.5,
        }
        ds = prepare_dataset(config=config)
        assert len(ds) == 100
        
        counts = {"isomorphic": 0, "non_isomorphic": 0}
        for item in ds:
            counts[item["task_type"]] += 1
            
        assert counts["isomorphic"] == 50
        assert counts["non_isomorphic"] == 50
