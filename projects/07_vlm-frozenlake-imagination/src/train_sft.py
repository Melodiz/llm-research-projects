"""Stage A4: LoRA SFT for FrozenLake world-model prediction"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

Condition = Literal["image_text", "image_only"]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DATA_DIR = Path("results/A3_dataset_collection")
DEFAULT_OUTPUT_ROOT = Path("results/A4_sft_training")

SYSTEM_PROMPT = (
    "You are a FrozenLake transition model. Predict the next player position "
    "and transition outcome from the rendered image and action. Return only "
    "the required prediction tag."
)

INSTRUCTION = (
    "Predict the next position and outcome for this deterministic 8x8 "
    "FrozenLake transition. Use zero-indexed (row, col) coordinates. "
    "Answer exactly in this format: "
    "<prediction>Position: (r, c). Outcome: safe</prediction>"
)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def user_text(row: dict[str, Any], condition: Condition) -> str:
    parts = [INSTRUCTION, f"Action: {row['action_name']}"]
    if condition == "image_text":
        parts.append(row["gt_state_text"])
    return "\n".join(parts)


def build_messages(row: dict[str, Any], condition: Condition, include_answer: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(row["image_path"]).convert("RGB")},
                {"type": "text", "text": user_text(row, condition)},
            ],
        },
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["target_text"]}],
            }
        )
    return messages


class FrozenLakeSFTDataset:
    def __init__(self, rows: list[dict[str, Any]], condition: Condition) -> None:
        self.rows = rows
        self.condition = condition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "row": row,
            "prompt_messages": build_messages(row, self.condition, include_answer=False),
            "full_messages": build_messages(row, self.condition, include_answer=True),
        }


@dataclass
class QwenVLSFTCollator:
    processor: Any
    process_vision_info: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        full_messages = [feature["full_messages"] for feature in features]
        prompt_messages = [feature["prompt_messages"] for feature in features]
        full_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in full_messages
        ]
        prompt_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompt_messages
        ]

        full_image_inputs, full_video_inputs = self.process_vision_info(full_messages)
        batch = self.processor(
            text=full_texts,
            images=full_image_inputs,
            videos=full_video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        # Mask all prompt tokens so loss is only on the assistant target text.
        for index, prompt_messages_one in enumerate(prompt_messages):
            prompt_image_inputs, prompt_video_inputs = self.process_vision_info([prompt_messages_one])
            prompt_batch = self.processor(
                text=[prompt_texts[index]],
                images=prompt_image_inputs,
                videos=prompt_video_inputs,
                return_tensors="pt",
            )
            prompt_len = int(prompt_batch["input_ids"].shape[1])
            labels[index, :prompt_len] = -100

        batch["labels"] = labels
        return batch


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_data_smoke(args: argparse.Namespace) -> None:
    train_rows = load_jsonl(args.data_dir / "train.jsonl", limit=args.train_limit)
    val_rows = load_jsonl(args.data_dir / "val.jsonl", limit=args.val_limit)
    examples = []
    print("Stage A4 data smoke")
    print(f"data_dir={args.data_dir}")
    print(f"condition={args.condition}")
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}")
    for row in train_rows[:3]:
        image = Image.open(row["image_path"])
        example = {
            "transition_id": row["transition_id"],
            "image_path": row["image_path"],
            "image_size": list(image.size),
            "action_name": row["action_name"],
            "target_text": row["target_text"],
            "user_text": user_text(row, args.condition),
        }
        examples.append(example)
        print(
            f"id={row['transition_id']} image={row['image_path']} "
            f"image_size={image.size} action={row['action_name']} target={row['target_text']}"
        )
        print(f"user_text={user_text(row, args.condition)}")
    smoke_path = args.output_root / "smoke" / f"{args.condition}_data_smoke.json"
    write_json(
        smoke_path,
        {
            "condition": args.condition,
            "data_dir": str(args.data_dir),
            "train_rows_loaded": len(train_rows),
            "val_rows_loaded": len(val_rows),
            "examples": examples,
        },
    )
    print(f"smoke_json={smoke_path}")


def load_model_and_processor(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=args.device_map,
        quantization_config=quantization_config,
    )
    return model, processor


def add_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def generate_samples(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    rows: list[dict[str, Any]],
    condition: Condition,
    output_path: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        messages = build_messages(row, condition, include_answer=False)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        decoded = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        samples.append(
            {
                "transition_id": row["transition_id"],
                "split": row["split"],
                "condition": condition,
                "action_name": row["action_name"],
                "target_text": row["target_text"],
                "generated_text": decoded,
                "image_path": row["image_path"],
                "gt_state_text": row["gt_state_text"],
            }
        )
    write_json(output_path, {"samples": samples})
    return samples


def run_train(args: argparse.Namespace) -> None:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import Trainer, TrainingArguments

    train_rows = load_jsonl(args.data_dir / "train.jsonl", limit=args.train_limit)
    val_rows = load_jsonl(args.data_dir / "val.jsonl", limit=args.val_limit)
    output_dir = args.output_root / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(args)
    model = add_lora(model, args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = FrozenLakeSFTDataset(train_rows, args.condition)
    val_dataset = FrozenLakeSFTDataset(val_rows, args.condition)
    collator = QwenVLSFTCollator(processor=processor, process_vision_info=process_vision_info)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=not args.bf16,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
    )

    config = {
        "condition": args.condition,
        "model_id": args.model_id,
        "method": "LoRA/QLoRA PEFT adapter",
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "bf16": args.bf16,
        "device_map": args.device_map,
        "output_dir": str(output_dir),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    write_json(output_dir / "config.json", config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    start_time = time.time()
    train_result = trainer.train()
    runtime_seconds = time.time() - start_time
    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "adapter"))
    write_json(output_dir / "train_result.json", train_result.metrics)
    write_json(output_dir / "trainer_log_history.json", {"log_history": trainer.state.log_history})

    train_samples = generate_samples(
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        rows=train_rows[: args.num_generate_samples],
        condition=args.condition,
        output_path=output_dir / "generated_train_samples.json",
        max_new_tokens=args.generation_max_new_tokens,
    )
    val_samples = generate_samples(
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        rows=val_rows[: args.num_generate_samples],
        condition=args.condition,
        output_path=output_dir / "generated_val_samples.json",
        max_new_tokens=args.generation_max_new_tokens,
    )

    final_payload = {
        "runtime_seconds": runtime_seconds,
        "train_metrics": train_result.metrics,
        "last_log": trainer.state.log_history[-1] if trainer.state.log_history else {},
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    write_json(output_dir / "run_summary.json", final_payload)
    print("Stage A4 SFT training complete")
    print(f"condition={args.condition}")
    print(f"output_dir={output_dir}")
    print(f"train_examples={len(train_rows)} val_examples={len(val_rows)}")
    print(f"runtime_seconds={runtime_seconds:.1f}")
    print(f"train_metrics={train_result.metrics}")
    print(f"adapter_dir={output_dir / 'adapter'}")


def run_generate_only(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    rows = load_jsonl(args.data_dir / "val.jsonl", limit=args.num_generate_samples)
    model, processor = load_model_and_processor(args)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    output_path = args.output_root / args.condition / "reload_generated_val_samples.json"
    samples = generate_samples(
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        rows=rows,
        condition=args.condition,
        output_path=output_path,
        max_new_tokens=args.generation_max_new_tokens,
    )
    print("Stage A4 adapter reload generation")
    print(f"condition={args.condition}")
    print(f"adapter_dir={args.adapter_dir}")
    for sample in samples:
        print(
            f"id={sample['transition_id']} target={sample['target_text']} "
            f"generated={sample['generated_text']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["data-smoke", "train", "generate-only"], default="data-smoke")
    parser.add_argument("--condition", choices=["image_text", "image_only"], default="image_text")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-generate-samples", type=int, default=3)
    parser.add_argument("--generation-max-new-tokens", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "data-smoke":
        run_data_smoke(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "generate-only":
        if args.adapter_dir is None:
            raise ValueError("--adapter-dir is required for --mode generate-only")
        run_generate_only(args)
    else:
        raise ValueError(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()
