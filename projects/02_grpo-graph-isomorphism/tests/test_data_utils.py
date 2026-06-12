"""Tests for the dataset preparation logic in training script."""

import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from train import prepare_dataset

def test_prepare_dataset_75_25_split():
    """Verify 75% iso, 25% non-iso split produces correct task_type assignments."""
    ds = prepare_dataset(num_questions=100, iso_ratio=0.75, difficulty=2, seed=42)
    
    assert len(ds) == 100
    
    # Check counts
    iso_count = sum(1 for row in ds if row["task_type"] == "isomorphic")
    noniso_count = sum(1 for row in ds if row["task_type"] == "non_isomorphic")
    
    assert iso_count == 75
    assert noniso_count == 25
    
    # Verify metadata matches task_type
    for row in ds:
        is_iso = row["metadata_list"]["is_isomorphic"]
        if row["task_type"] == "isomorphic":
            assert is_iso is True
        else:
            assert is_iso is False
            
def test_prepare_dataset_50_50_split():
    """Verify 50% iso, 50% non-iso split produces correct task_type assignments."""
    ds = prepare_dataset(num_questions=20, iso_ratio=0.5, difficulty=2, seed=42)
    
    assert len(ds) == 20
    
    # Check counts
    iso_count = sum(1 for row in ds if row["task_type"] == "isomorphic")
    noniso_count = sum(1 for row in ds if row["task_type"] == "non_isomorphic")
    
    assert iso_count == 10
    assert noniso_count == 10
    
    # Verify metadata matches task_type
    for row in ds:
        is_iso = row["metadata_list"]["is_isomorphic"]
        if row["task_type"] == "isomorphic":
            assert is_iso is True
        else:
            assert is_iso is False
