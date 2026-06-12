#!/usr/bin/env python3
"""Generate frozen test sets for Graph Isomorphism evaluation."""

import sys
import os
import json
import hashlib
from datetime import datetime, timezone

# Ensure src/ (this file's directory) is importable; PROJECT_ROOT is its parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from graph_isomorphism_env import GraphIsomorphismEnv, DIFFICULTY_MAP


MIXED_SEED = 9999
ISO_ONLY_SEED = 9998
HINT_SEED = 9997
NON_ISO_SEED = 9996
DIFFICULTIES = range(1, 11)
HINT_DIFFICULTIES = [3, 4, 5]
MIXED_COUNT = 300       # per difficulty
ISO_ONLY_COUNT = 150    # per difficulty
NON_ISO_COUNT = 50      # per difficulty per category
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "test_sets")


def _data_to_dict(d):
    """Convert Data instance to dict."""
    meta = dict(d.metadata)
    # ground_truth_perm has int keys — JSON needs string keys
    if "ground_truth_perm" in meta:
        meta["ground_truth_perm"] = {
            str(k): v for k, v in meta["ground_truth_perm"].items()
        }
    return {
        "question": d.question,
        "answer": d.answer,
        "difficulty": d.difficulty,
        "metadata": meta,
    }


def _write_jsonl(data_list, path):
    """Write list of Data objects as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for d in data_list:
            f.write(json.dumps(_data_to_dict(d), sort_keys=True) + "\n")
    return len(data_list)


def _file_sha256(path):
    """Compute SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_source_hash():
    """Hash of graph_isomorphism_env.py."""
    env_path = os.path.join(PROJECT_ROOT, "graph_isomorphism_env.py")
    return _file_sha256(env_path)


def generate_all():
    env = GraphIsomorphismEnv()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env_source_hash": _env_source_hash(),
        "mixed_seed": MIXED_SEED,
        "iso_only_seed": ISO_ONLY_SEED,
        "hint_seed": HINT_SEED,
        "non_iso_seed": NON_ISO_SEED,
        "files": {},
    }

    total_mixed = 0
    total_iso = 0
    total_hints = 0
    total_non_iso = 0


    print("Generating MIXED test sets (50/50 iso/non-iso)...")
    for diff in DIFFICULTIES:
        data = env.generate(
            num_of_questions=MIXED_COUNT,
            difficulty=diff,
            iso_ratio=0.5,
            seed=MIXED_SEED + diff,  # unique per difficulty, deterministic
        )
        fname = f"mixed_diff{diff}.jsonl"
        path = os.path.join(OUTPUT_DIR, fname)
        n = _write_jsonl(data, path)
        total_mixed += n

        iso_count = sum(1 for d in data if d.metadata["is_isomorphic"])
        manifest["files"][fname] = {
            "count": n,
            "iso_count": iso_count,
            "noniso_count": n - iso_count,
            "difficulty": diff,
            "seed": MIXED_SEED + diff,
            "sha256": _file_sha256(path),
        }
        print(f"  diff={diff:2d}: {n} instances ({iso_count} iso, {n - iso_count} non-iso)")


    print("\nGenerating ISO-ONLY test sets...")
    for diff in DIFFICULTIES:
        data = env.generate(
            num_of_questions=ISO_ONLY_COUNT,
            difficulty=diff,
            iso_ratio=1.0,
            seed=ISO_ONLY_SEED + diff,
        )
        fname = f"iso_only_diff{diff}.jsonl"
        path = os.path.join(OUTPUT_DIR, fname)
        n = _write_jsonl(data, path)
        total_iso += n

        manifest["files"][fname] = {
            "count": n,
            "iso_count": n,
            "noniso_count": 0,
            "difficulty": diff,
            "seed": ISO_ONLY_SEED + diff,
            "sha256": _file_sha256(path),
        }
        print(f"  diff={diff:2d}: {n} instances (all iso)")


    print("\nGenerating HINTS test sets (k=n-2)...")
    for diff in HINT_DIFFICULTIES:
        n_nodes = DIFFICULTY_MAP[diff]["n_nodes"]
        k = max(1, n_nodes - 2)
        
        data = env.generate(
            num_of_questions=ISO_ONLY_COUNT,
            difficulty=diff,
            iso_ratio=1.0,  # Hints are only for iso pairs
            seed=HINT_SEED + diff,
            hints_k=k,
        )
        fname = f"hints_k{k}_diff{diff}.jsonl"
        path = os.path.join(OUTPUT_DIR, fname)
        n = _write_jsonl(data, path)
        total_hints += n

        manifest["files"][fname] = {
            "count": n,
            "iso_count": n,
            "noniso_count": 0,
            "difficulty": diff,
            "hints_k": k,
            "seed": HINT_SEED + diff,
            "sha256": _file_sha256(path),
        }
        print(f"  diff={diff:2d}: {n} instances (k={k})")


    print("\nGenerating NON-ISO stratified test sets (easy/medium/hard)...")
    for category in ["easy", "medium", "hard"]:
        for diff in DIFFICULTIES:
            seed = NON_ISO_SEED + diff + {"easy": 0, "medium": 100, "hard": 200}[category]
            data = env.generate(
                num_of_questions=NON_ISO_COUNT,
                difficulty=diff,
                iso_ratio=0.0,  # All non-isomorphic
                non_iso=category,
                seed=seed,
            )
            fname = f"non_iso_{category}_diff{diff}.jsonl"
            path = os.path.join(OUTPUT_DIR, fname)
            n = _write_jsonl(data, path)
            total_non_iso += n

            manifest["files"][fname] = {
                "count": n,
                "iso_count": 0,
                "noniso_count": n,
                "difficulty": diff,
                "non_iso_category": category,
                "seed": seed,
                "sha256": _file_sha256(path),
            }
            print(f"  {category:6s} diff={diff:2d}: {n} instances")


    manifest["total_mixed"] = total_mixed
    manifest["total_iso_only"] = total_iso
    manifest["total_hints"] = total_hints
    manifest["total_non_iso"] = total_non_iso
    manifest["total"] = total_mixed + total_iso + total_hints + total_non_iso

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\n{'='*50}")
    print(f"Mixed:    {total_mixed} instances (10 files)")
    print(f"ISO-only: {total_iso} instances (10 files)")
    print(f"Hints:    {total_hints} instances ({len(HINT_DIFFICULTIES)} files)")
    print(f"Non-iso:  {total_non_iso} instances (30 files)")
    print(f"Total:    {manifest['total']} instances")
    print(f"Manifest: {manifest_path}")
    print(f"Env hash: {manifest['env_source_hash'][:16]}...")

    return manifest


if __name__ == "__main__":
    generate_all()
