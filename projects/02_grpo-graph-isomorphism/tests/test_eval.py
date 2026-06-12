"""Unit tests for eval.py metric computation logic."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eval import compute_metrics


def _wrap(ans):
    """Helper to simulate valid reasoning format."""
    return f"<think>thinking</think>\n<answer>{ans}</answer>"


def test_empty_results():
    assert compute_metrics([], [], []) == {}


class TestMetrics:
    def test_all_correct_perfect_score(self):
        """1 iso, 1 non-iso. Both correct and well-formatted."""
        results = [
            {
                "difficulty": 1,
                "is_isomorphic": True,
                "question": "q",
                "ground_truth": "1->2, 2->1",
                "completion": _wrap("1->2, 2->1"),
                "metadata": {"is_isomorphic": True, "n_nodes": 2, "g1_edges": [[1,2]], "g2_edges": [[1,2]]}
            },
            {
                "difficulty": 1,
                "is_isomorphic": False,
                "question": "q",
                "ground_truth": "NOT ISOMORPHIC",
                "completion": _wrap("NOT ISOMORPHIC"),
                "metadata": {"is_isomorphic": False, "n_nodes": 2, "g1_edges": [[1,2]], "g2_edges": []}
            }
        ]
        
        c = [r["completion"] for r in results]
        g = [r["ground_truth"] for r in results]
        m = [r["metadata"] for r in results]
        mets = compute_metrics(c, g, m)
        
        assert mets["total"] == 2
        assert mets["iso_count"] == 1
        assert mets["noniso_count"] == 1
        
        assert mets["aggregate_accuracy"] == 1.0
        assert mets["iso_accuracy"] == 1.0
        assert mets["non_iso_accuracy"] == 1.0
        assert mets["class_prediction_ratio"] == 0.5  # 1 out of 2 said NOT ISO
        assert mets["format_compliance"] == 1.0
        
        # Taxonomy should be zero
        assert mets["err_format_fail"] == 0
        assert mets["err_wrong_mapping"] == 0
        assert mets["err_false_not_iso"] == 0
        assert mets["err_false_mapping_claim"] == 0
        
        assert mets["degenerate_floor"] == 0.5  # 1 non-iso / 2 total

    def test_degenerate_collapse_collapse(self):
        """Model always says NOT ISOMORPHIC. (75 iso, 25 noniso split)."""
        results = []
        
        # 75 iso that it gets WRONG by saying NOT ISO
        for _ in range(75):
            results.append({
                "difficulty": 1,
                "is_isomorphic": True,
                "question": "q",
                "ground_truth": "1->1",
                "completion": _wrap("NOT ISOMORPHIC"),
                "metadata": {"is_isomorphic": True, "n_nodes": 1, "g1_edges": [], "g2_edges": []}
            })
            
        # 25 non-iso that it gets RIGHT
        for _ in range(25):
            results.append({
                "difficulty": 1,
                "is_isomorphic": False,
                "question": "q",
                "ground_truth": "NOT ISOMORPHIC",
                "completion": _wrap("NOT ISOMORPHIC"),
                "metadata": {"is_isomorphic": False, "n_nodes": 1, "g1_edges": [], "g2_edges": []}
            })
            
        c = [r["completion"] for r in results]
        g = [r["ground_truth"] for r in results]
        m = [r["metadata"] for r in results]
        mets = compute_metrics(c, g, m)
        
        assert mets["aggregate_accuracy"] == 0.25 # Only 25/100 correct
        assert mets["iso_accuracy"] == 0.0        # Failed all 75
        assert mets["non_iso_accuracy"] == 1.0    # Passed all 25
        assert mets["class_prediction_ratio"] == 1.0 # Said NOT ISO 100 times
        
        # Degenerate floor = 25 / 100 = 0.25.
        # This shows it learned nothing beyond the floor.
        assert mets["degenerate_floor"] == 0.25   
        
        # All 75 iso failures are because it falsely claimed NOT ISO
        assert mets["err_false_not_iso"] == 75
        assert mets["err_wrong_mapping"] == 0

    def test_error_taxonomy_breakdown(self):
        """Test the 4 specific error categories."""
        results = [
            # 1. Format fail on iso
            {
                "difficulty": 1,
                "is_isomorphic": True,
                "question": "q",
                "ground_truth": "1->1",
                "completion": "I forgot the answer tags but 1->1",
                "metadata": {"is_isomorphic": True, "n_nodes": 1, "g1_edges": [], "g2_edges": []}
            },
            # 2. Format fail on non-iso
            {
                "difficulty": 1,
                "is_isomorphic": False,
                "question": "q",
                "ground_truth": "NOT ISOMORPHIC",
                "completion": "Nothing here...",
                "metadata": {"is_isomorphic": False, "n_nodes": 1, "g1_edges": [], "g2_edges": []}
            },
            # 3. Wrong mapping on iso
            {
                "difficulty": 1,
                "is_isomorphic": True,
                "question": "q",
                "ground_truth": "1->2, 2->1",
                "completion": _wrap("1->1, 2->2"), # Wrong bijection
                "metadata": {"is_isomorphic": True, "n_nodes": 2, "g1_edges": [[1,2]], "g2_edges": []}
            },
            # 4. False NOT ISO on iso
            {
                "difficulty": 1,
                "is_isomorphic": True,
                "question": "q",
                "ground_truth": "1->1",
                "completion": _wrap("NOT ISOMORPHIC"),
                "metadata": {"is_isomorphic": True, "n_nodes": 1, "g1_edges": [], "g2_edges": []}
            },
            # 5. False mapping claim on non-iso
            {
                "difficulty": 1,
                "is_isomorphic": False,
                "question": "q",
                "ground_truth": "NOT ISOMORPHIC",
                "completion": _wrap("1->1"), # Tried to map an impossible graph
                "metadata": {"is_isomorphic": False, "n_nodes": 1, "g1_edges": [], "g2_edges": []}
            }
        ]
        
        c = [r["completion"] for r in results]
        g = [r["ground_truth"] for r in results]
        m = [r["metadata"] for r in results]
        mets = compute_metrics(c, g, m)
        
        assert mets["err_format_fail"] == 2 # Case 1 and 2
        assert mets["err_wrong_mapping"] == 1 # Case 3
        assert mets["err_false_not_iso"] == 1 # Case 4
        assert mets["err_false_mapping_claim"] == 1 # Case 5
        
        assert mets["format_compliance"] == 0.6 # 3 out of 5 had valid format tags
