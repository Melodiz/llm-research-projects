"""Tests for rewards.py — class_aware_reward and composite_reward."""

import sys
import os
import pytest
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rewards import class_aware_reward, composite_reward, _extract_text
from graph_isomorphism_env import (
    GraphIsomorphismEnv,
    GraphIsomorphismVerifier,
    Data,
    _edges_to_sorted_tuples,
    _generate_isomorphic_pair,
)
import random




def _make_meta(is_iso: bool, G1, G2, perm=None):
    n = G1.number_of_nodes()
    meta = {
        "is_isomorphic": is_iso,
        "n_nodes": n,
        "g1_edges": _edges_to_sorted_tuples(G1),
        "g2_edges": _edges_to_sorted_tuples(G2),
    }
    if perm:
        meta["ground_truth_perm"] = perm
    return meta


def _make_iso_instance(seed=42):
    """Return (ground_truth_str, metadata, task_type) for one iso pair."""
    rng = random.Random(seed)
    G1, G2, perm, perm_str = _generate_isomorphic_pair(4, 0.5, rng)
    meta = _make_meta(True, G1, G2, perm)
    return perm_str, meta, "isomorphic"


def _make_noniso_instance():
    """Return (ground_truth_str, metadata, task_type) for one non-iso pair."""
    G1 = nx.Graph()
    G1.add_nodes_from([1, 2, 3, 4])
    G1.add_edges_from([(1, 2), (2, 3)])
    G2 = nx.Graph()
    G2.add_nodes_from([1, 2, 3, 4])
    G2.add_edges_from([(1, 2), (3, 4)])
    meta = _make_meta(False, G1, G2)
    return "NOT ISOMORPHIC", meta, "non_isomorphic"


def _wrap(answer_text):
    return f"<think>reasoning</think><answer>{answer_text}</answer>"


def _wrap_trl(answer_text):
    """trl >=0.16 format."""
    return [{"role": "assistant", "content": _wrap(answer_text)}]




class TestAlwaysNotIsoDegenerate:
    def _build_batch(self, n_iso: int, n_noniso: int):
        """Build a batch where the model always says NOT ISOMORPHIC."""
        completions = []
        task_types = []
        ground_truths = []
        metadata_list = []

        for i in range(n_iso):
            gt, meta, tt = _make_iso_instance(seed=100 + i)
            completions.append(_wrap("NOT ISOMORPHIC"))
            task_types.append(tt)
            ground_truths.append(gt)
            metadata_list.append(meta)

        for i in range(n_noniso):
            gt, meta, tt = _make_noniso_instance()
            completions.append(_wrap("NOT ISOMORPHIC"))
            task_types.append(tt)
            ground_truths.append(gt)
            metadata_list.append(meta)

        return completions, task_types, ground_truths, metadata_list

    def test_class_aware_negative_mean_75_25(self):
        """Check negative mean reward for degenerate strategy."""
        comps, tts, gts, metas = self._build_batch(75, 25)
        rewards = class_aware_reward(comps, tts, gts, metas)
        mean_r = sum(rewards) / len(rewards)
        # Expected: 0.75 * (-1.0 + 0.1) + 0.25 * (0.5 + 0.1)
        #         = 0.75 * (-0.9) + 0.25 * 0.6
        #         = -0.675 + 0.15 = -0.525
        assert mean_r < 0, f"Mean reward should be negative, got {mean_r:.4f}"
        assert abs(mean_r - (-0.525)) < 0.01, f"Expected ≈ -0.525, got {mean_r:.4f}"

    def test_composite_positive_mean_50_50(self):
        """Check positive mean reward for degenerate strategy."""
        comps, tts, gts, metas = self._build_batch(50, 50)
        # composite_reward doesn't use task_type
        rewards = composite_reward(comps, gts, metas)
        mean_r = sum(rewards) / len(rewards)
        # Expected: 0.5 * (0.0 + 0.1) + 0.5 * (1.0 + 0.1)
        #         = 0.5 * 0.1 + 0.5 * 1.1 = 0.05 + 0.55 = 0.6
        assert mean_r > 0, f"Mean reward should be positive, got {mean_r:.4f}"
        assert abs(mean_r - 0.6) < 0.01, f"Expected ≈ 0.6, got {mean_r:.4f}"




class TestExactRewardValues:
    def test_iso_correct_with_format(self):
        """Iso correct + format → 1.5 + 0.1 = 1.6."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap(gt)  # correct mapping in proper format
        rewards = class_aware_reward([comp], [tt], [gt], [meta])
        assert rewards[0] == pytest.approx(1.6), f"Got {rewards[0]}"

    def test_noniso_correct_with_format(self):
        """Non-iso correct + format → 0.5 + 0.1 = 0.6."""
        gt, meta, tt = _make_noniso_instance()
        comp = _wrap("NOT ISOMORPHIC")
        rewards = class_aware_reward([comp], [tt], [gt], [meta])
        assert rewards[0] == pytest.approx(0.6), f"Got {rewards[0]}"

    def test_iso_wrong_with_format(self):
        """Iso wrong + format → -1.0 + 0.1 = -0.9."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap("NOT ISOMORPHIC")  # wrong for iso pair
        rewards = class_aware_reward([comp], [tt], [gt], [meta])
        assert rewards[0] == pytest.approx(-0.9), f"Got {rewards[0]}"

    def test_noniso_wrong_with_format(self):
        """Non-iso wrong + format → -0.5 + 0.1 = -0.4."""
        gt, meta, tt = _make_noniso_instance()
        # Provide a (wrong) mapping for non-iso pair
        comp = _wrap("1->1, 2->2, 3->3, 4->4")
        rewards = class_aware_reward([comp], [tt], [gt], [meta])
        assert rewards[0] == pytest.approx(-0.4), f"Got {rewards[0]}"

    def test_iso_correct_no_format(self):
        """Iso correct without proper format → 1.5 + 0.0 = 1.5."""
        gt, meta, tt = _make_iso_instance()
        # answer tags but no think tags → format bonus = 0
        comp = f"<answer>{gt}</answer>"
        rewards = class_aware_reward([comp], [tt], [gt], [meta])
        assert rewards[0] == pytest.approx(1.5), f"Got {rewards[0]}"

    def test_composite_correct_with_format(self):
        """Composite: correct + format → 1.0 + 0.1 = 1.1."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap(gt)
        rewards = composite_reward([comp], [gt], [meta])
        assert rewards[0] == pytest.approx(1.1), f"Got {rewards[0]}"

    def test_composite_wrong_with_format(self):
        """Composite: wrong + format → 0.0 + 0.1 = 0.1."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap("NOT ISOMORPHIC")
        rewards = composite_reward([comp], [gt], [meta])
        assert rewards[0] == pytest.approx(0.1), f"Got {rewards[0]}"




class TestCompletionFormats:
    def test_extract_text_string(self):
        assert _extract_text("hello") == "hello"

    def test_extract_text_list_of_dicts(self):
        comp = [{"role": "assistant", "content": "hello"}]
        assert _extract_text(comp) == "hello"

    def test_extract_text_empty(self):
        assert _extract_text("") == ""
        assert _extract_text(None) == ""
        assert _extract_text([]) == ""

    def test_class_aware_with_trl_format(self):
        """class_aware_reward works with list-of-dicts completions."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap_trl(gt)  # trl >=0.16 format
        rewards = class_aware_reward([comp], [tt], [gt], [meta])
        assert rewards[0] == pytest.approx(1.6), f"Got {rewards[0]}"

    def test_composite_with_trl_format(self):
        """composite_reward works with list-of-dicts completions."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap_trl(gt)
        rewards = composite_reward([comp], [gt], [meta])
        assert rewards[0] == pytest.approx(1.1), f"Got {rewards[0]}"

    def test_class_aware_with_string_format(self):
        """class_aware_reward works with plain string completions."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap(gt)  # plain string
        rewards = class_aware_reward([comp], [tt], [gt], [meta])
        assert rewards[0] == pytest.approx(1.6), f"Got {rewards[0]}"




class TestKwargsInterface:
    def test_task_type_as_list_of_strings(self):
        """task_type passed as list of strings — must not crash."""
        gt_iso, meta_iso, tt_iso = _make_iso_instance()
        gt_noniso, meta_noniso, tt_noniso = _make_noniso_instance()

        comps = [_wrap(gt_iso), _wrap("NOT ISOMORPHIC")]
        tts = [tt_iso, tt_noniso]
        gts = [gt_iso, gt_noniso]
        metas = [meta_iso, meta_noniso]

        rewards = class_aware_reward(comps, tts, gts, metas)
        assert len(rewards) == 2
        # First: iso correct with format = 1.6
        assert rewards[0] == pytest.approx(1.6)
        # Second: non-iso correct with format = 0.6
        assert rewards[1] == pytest.approx(0.6)

    def test_extra_kwargs_ignored(self):
        """Extra kwargs should be silently ignored."""
        gt, meta, tt = _make_iso_instance()
        comp = _wrap(gt)
        # Pass extra kwargs that don't exist in the signature
        rewards = class_aware_reward(
            [comp], [tt], [gt], [meta],
            some_random_key="value", another_key=42,
        )
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(1.6)
