"""SFT Warmup for Run S1 — Phase A."""

import os
import json
import random
import argparse
import networkx as nx
from graph_isomorphism_env import (
    GraphIsomorphismEnv, _graph_to_text, _edges_to_sorted_tuples,
)



def _build_iso_cot(G1: nx.Graph, G2: nx.Graph, perm: dict) -> str:
    """Build a full chain-of-thought response for an isomorphic pair."""
    n = G1.number_of_nodes()
    m1 = G1.number_of_edges()
    m2 = G2.number_of_edges()

    deg_seq_g1 = sorted([d for _, d in G1.degree()], reverse=True)
    deg_seq_g2 = sorted([d for _, d in G2.degree()], reverse=True)

    lines = []
    lines.append(f"G1 and G2 both have {n} nodes and {m1} edges.")
    lines.append(f"Degree sequence of G1: {deg_seq_g1}.")
    lines.append(f"Degree sequence of G2: {deg_seq_g2}.")

    if deg_seq_g1 == deg_seq_g2:
        lines.append("The degree sequences match, so they could be isomorphic.")
    else:
        # Shouldn't happen for iso pairs, but handle gracefully
        lines.append("Degree sequences differ — but let me check the mapping anyway.")

    lines.append("")
    lines.append("Trying mapping: " + ", ".join(
        f"node {k}→{v}" for k, v in sorted(perm.items())
    ))

    # Verify each edge
    g2_edges = set()
    for u, v in G2.edges():
        g2_edges.add((min(u, v), max(u, v)))

    all_ok = True
    for u, v in sorted(G1.edges()):
        mu, mv = perm[u], perm[v]
        mapped_edge = (min(mu, mv), max(mu, mv))
        if mapped_edge in g2_edges:
            lines.append(f"Edge ({u},{v}) in G1 → edge ({mu},{mv}) in G2 ✓")
        else:
            lines.append(f"Edge ({u},{v}) in G1 → edge ({mu},{mv}) NOT in G2 ✗")
            all_ok = False

    if all_ok:
        lines.append("All edges verified — the graphs are isomorphic.")

    cot = "\n".join(lines)
    answer = ", ".join(f"{k}->{v}" for k, v in sorted(perm.items()))
    return f"<think>{cot}</think>\n<answer>{answer}</answer>"


def _build_non_iso_cot(G1: nx.Graph, G2: nx.Graph) -> str:
    """Build a chain-of-thought response for a non-isomorphic pair."""
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    m1 = G1.number_of_edges()
    m2 = G2.number_of_edges()

    deg_seq_g1 = sorted([d for _, d in G1.degree()], reverse=True)
    deg_seq_g2 = sorted([d for _, d in G2.degree()], reverse=True)

    lines = []

    if n1 != n2:
        lines.append(f"G1 has {n1} nodes but G2 has {n2} nodes.")
        lines.append("Different number of nodes means they cannot be isomorphic.")
    elif m1 != m2:
        lines.append(f"G1 has {n1} nodes and {m1} edges.")
        lines.append(f"G2 has {n2} nodes and {m2} edges.")
        lines.append("Different number of edges means they cannot be isomorphic.")
    elif deg_seq_g1 != deg_seq_g2:
        lines.append(f"G1 has {n1} nodes and {m1} edges.")
        lines.append(f"G2 has {n2} nodes and {m2} edges.")
        lines.append(f"Degree sequence of G1: {deg_seq_g1}.")
        lines.append(f"Degree sequence of G2: {deg_seq_g2}.")
        lines.append("The degree sequences differ, so the graphs are not isomorphic.")
    else:
        lines.append(f"Both graphs have {n1} nodes and {m1} edges.")
        lines.append(f"Degree sequences both: {deg_seq_g1}.")
        lines.append("Despite same degree sequences, no valid mapping exists.")
        lines.append("Checked all possible mappings — none preserve all edges.")

    cot = "\n".join(lines)
    return f"<think>{cot}</think>\n<answer>NOT ISOMORPHIC</answer>"


def generate_sft_examples(
    num_iso: int = 30,
    num_non_iso: int = 20,
    difficulties: list = None,
    seed: int = 42,
) -> list:
    """Generate SFT training examples with full CoT."""
    if difficulties is None:
        difficulties = [1, 2]

    SYSTEM_PROMPT = (
        "Respond in the following format:\n"
        "<think>\n...\n</think>\n"
        "<answer>\n...\n</answer>"
    )

    env = GraphIsomorphismEnv()
    rng = random.Random(seed)
    examples = []

    # Split counts across difficulties
    iso_per_diff = max(1, num_iso // len(difficulties))
    non_iso_per_diff = max(1, num_non_iso // len(difficulties))

    for diff in difficulties:
        # Iso examples
        iso_data = env.generate(
            num_of_questions=iso_per_diff,
            difficulty=diff, iso_ratio=1.0,
            seed=rng.randint(0, 2**31),
        )
        for d in iso_data:
            perm = d.metadata["ground_truth_perm"]
            # Reconstruct graphs from edges for CoT
            n = d.metadata["n_nodes"]
            G1 = nx.Graph()
            G1.add_nodes_from(range(1, n + 1))
            G1.add_edges_from(d.metadata["g1_edges"])
            G2 = nx.Graph()
            G2.add_nodes_from(range(1, n + 1))
            G2.add_edges_from(d.metadata["g2_edges"])

            response = _build_iso_cot(G1, G2, perm)
            examples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d.question},
                ],
                "response": response,
                "task_type": "isomorphic",
            })

        # Non-iso examples
        non_iso_data = env.generate(
            num_of_questions=non_iso_per_diff,
            difficulty=diff, iso_ratio=0.0,
            seed=rng.randint(0, 2**31),
        )
        for d in non_iso_data:
            n = d.metadata["n_nodes"]
            G1 = nx.Graph()
            G1.add_nodes_from(range(1, n + 1))
            G1.add_edges_from(d.metadata["g1_edges"])
            g2_n = d.metadata.get("g2_n", n)
            G2 = nx.Graph()
            G2.add_nodes_from(range(1, g2_n + 1))
            G2.add_edges_from(d.metadata["g2_edges"])

            response = _build_non_iso_cot(G1, G2)
            examples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d.question},
                ],
                "response": response,
                "task_type": "non_isomorphic",
            })

    rng.shuffle(examples)
    return examples


def examples_to_chat_format(examples: list) -> list:
    """Convert examples to chat-format dicts for SFTTrainer."""
    formatted = []
    for ex in examples:
        messages = list(ex["prompt"]) + [
            {"role": "assistant", "content": ex["response"]},
        ]
        formatted.append({"messages": messages})
    return formatted



def run_sft(
    examples: list,
    output_dir: str = "outputs/run_s1_sft",
    max_steps: int = 20,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
):
    """Phase A: SFT training on CoT examples using unsloth."""
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    os.environ['WANDB_MODE'] = 'disabled'
    os.makedirs(output_dir, exist_ok=True)

    # Same model loading as train.py — no vLLM for SFT (omit fast_inference)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=1024,
        load_in_4bit=True,
        max_lora_rank=32,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Prepare dataset — pre-apply chat template into a "text" column.
    # This is the most reliable path with unsloth's SFTTrainer:
    # avoids formatting_func batching issues entirely.
    chat_data = examples_to_chat_format(examples)
    for row in chat_data:
        row["text"] = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )
    dataset = Dataset.from_list(chat_data)

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=1,
        save_steps=max_steps,   # save at end
        bf16=True,
        max_seq_length=1024,
        packing=False,
        dataset_text_field="text",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print(f"=== SFT Warmup: {len(examples)} examples, {max_steps} steps ===")
    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"=== SFT adapter saved to {output_dir} ===")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--num_iso", type=int, default=30)
    parser.add_argument("--num_non_iso", type=int, default=20)
    parser.add_argument("--difficulties", type=str, default="1,2")
    parser.add_argument("--sft_steps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs/run_s1_sft")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    difficulties = [int(d) for d in args.difficulties.split(",")]

    print(f"Generating {args.num_iso} iso + {args.num_non_iso} non-iso CoT examples...")
    examples = generate_sft_examples(
        num_iso=args.num_iso,
        num_non_iso=args.num_non_iso,
        difficulties=difficulties,
        seed=args.seed,
    )

    # Dump to JSON for inspection
    examples_path = os.path.join(args.output_dir, "sft_examples.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} examples to {examples_path}")

    iso_count = sum(1 for ex in examples if ex["task_type"] == "isomorphic")
    print(f"  Iso: {iso_count}, Non-iso: {len(examples) - iso_count}")

    # Show a sample
    if examples:
        sample = examples[0]
        print(f"\n--- Sample ({sample['task_type']}) ---")
        print(sample["response"][:300])
        print("...")

    if args.generate_only:
        print("\n--generate-only: skipping SFT training.")
        return

    run_sft(
        examples=examples,
        output_dir=args.output_dir,
        max_steps=args.sft_steps,
    )


if __name__ == "__main__":
    main()
