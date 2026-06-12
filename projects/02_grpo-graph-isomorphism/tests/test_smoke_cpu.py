"""End-to-end CPU smoke test for the Graph Isomorphism GRPO pipeline."""

import os
import sys
import tempfile
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from data_utils import prepare_dataset
from rewards import class_aware_reward, composite_reward
from eval import compute_metrics


def _wrap(ans):
    return f"<think>thinking</think>\n<answer>{ans}</answer>"


class TestCPUPipeline:
    def test_end_to_end_pipeline(self):

        ds = prepare_dataset(
            difficulty=1,
            iso_ratio=0.75,
            num_questions=20,
            seed=42
        )
        
        assert len(ds) == 20
        assert "prompt" in ds.column_names
        assert "ground_truth" in ds.column_names
        assert "metadata_list" in ds.column_names
        assert "task_type" in ds.column_names
        
        # Verify 75/25 split (15 iso, 5 non-iso)
        iso_count = sum(1 for x in ds["task_type"] if x == "isomorphic")
        noniso_count = sum(1 for x in ds["task_type"] if x == "non_isomorphic")
        assert iso_count == 15
        assert noniso_count == 5


        # Simulate completions: 
        # Half of the iso get it right, half say NOT ISO
        # All non-iso say NOT ISO
        completions = []
        expected_iso_correct = 0
        for i, row in enumerate(ds):
            if row["task_type"] == "isomorphic":
                if expected_iso_correct < 7: # Force exactly 7 correct
                    completions.append(_wrap(row["ground_truth"]))  # Correct
                    expected_iso_correct += 1
                else:
                    completions.append(_wrap("NOT ISOMORPHIC"))     # Wrong
            else:
                completions.append(_wrap("NOT ISOMORPHIC"))         # Correct
        
        # Test class_aware_reward
        ca_rewards = class_aware_reward(
            completions=completions,
            task_type=ds["task_type"],
            ground_truth=ds["ground_truth"],
            metadata_list=ds["metadata_list"]
        )
        
        assert len(ca_rewards) == 20
        # Iso correct (split exactly 7 correct, 8 wrong because 15 iso)
        for i, (r, tt, c, gt) in enumerate(zip(ca_rewards, ds["task_type"], completions, ds["ground_truth"])):
            if tt == "isomorphic":
                if "NOT ISOMORPHIC" in c:
                    assert r == pytest.approx(-0.9) # -1.0 + 0.1
                else:
                    assert r == pytest.approx(1.6) # 1.5 + 0.1
            else:
                assert r == pytest.approx(0.6) # 0.5 + 0.1

        # Test composite_reward
        comp_rewards = composite_reward(
            completions=completions,
            ground_truth=ds["ground_truth"],
            metadata_list=ds["metadata_list"]
        )
        assert len(comp_rewards) == 20


        degen_comps = [_wrap("NOT ISOMORPHIC")] * 20
        
        # 75/25 split (class aware must be negative)
        degen_ca = class_aware_reward(
            degen_comps, ds["task_type"], ds["ground_truth"], ds["metadata_list"]
        )
        ca_mean = sum(degen_ca) / 20
        assert ca_mean < 0, f"class_aware_reward failed anti-hacking check. Mean: {ca_mean}"

        # 50/50 split (composite must be positive)
        ds50 = prepare_dataset(difficulty=1, iso_ratio=0.5, num_questions=20, seed=42)
        degen_comps50 = [_wrap("NOT ISOMORPHIC")] * 20
        degen_comp50 = composite_reward(
            degen_comps50, ds50["ground_truth"], ds50["metadata_list"]
        )
        comp_mean = sum(degen_comp50) / 20
        assert comp_mean > 0, f"composite_reward failed positive check. Mean: {comp_mean}"


        metrics = compute_metrics(
            completions=completions,
            ground_truths=ds["ground_truth"],
            metadata_list=ds["metadata_list"]
        )
        
        assert metrics["total"] == 20
        assert metrics["iso_count"] == 15
        assert metrics["noniso_count"] == 5
        assert metrics["format_compliance"] == 1.0
        
        # 7 correct iso, 5 correct non-iso = 12/20
        assert metrics["iso_accuracy"] == 7 / 15
        assert metrics["non_iso_accuracy"] == 1.0
        assert metrics["aggregate_accuracy"] == 12 / 20
        
        # Class prediction: 8 false NOT ISO + 5 true NOT ISO = 13 / 20
        assert metrics["class_prediction_ratio"] == 13 / 20
        
        # Degenerate floor: 5 non-iso / 20 total = 0.25
        assert metrics["degenerate_floor"] == 0.25
        
        # Error taxonomy: all 8 failures on iso were false NOT ISO
        assert metrics["err_false_not_iso"] == 8
        assert metrics["err_wrong_mapping"] == 0
        assert metrics["err_format_fail"] == 0
        assert metrics["err_false_mapping_claim"] == 0
