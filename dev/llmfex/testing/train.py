#!/usr/bin/env python3
"""
train_pde_llm.py

Minimal, practical fine-tuning script for:
  - meta-llama/Llama-3.2-3B (3B base)
  - QLoRA (4-bit NF4)
  - LoRA adapters for attention projections
  - Trainer-based fine-tuning (Hugging Face Transformers + PEFT)
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def read_jsonl(path: str) -> List[Dict]:
    data =[]
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_dataset(path: str, val_split: float = 0.05, seed: int = 42):
    # Accept JSONL (one JSON per line) or a single JSON list
    if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
        data = read_jsonl(path)
    else:
        # try load as a single json list
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            data = raw
        else:
            raise ValueError("Unsupported JSON format. Expecting JSONL or list-of-objects JSON.")
            
    # Normalize field names: accept "prompt" or "input"
    norm =[]
    for i, item in enumerate(data):
        if "prompt" in item:
            prompt = item["prompt"]
        elif "input" in item:
            prompt = item["input"]
        else:
            raise ValueError(f"Item {i} missing 'prompt' or 'input' field")
        if "target" not in item:
            raise ValueError(f"Item {i} missing 'target' field")
        target = item["target"]
        norm.append({"prompt": prompt, "target": target})
        
    full = Dataset.from_list(norm)
    if val_split > 0:
        split = full.train_test_split(test_size=val_split, seed=seed)
        return split["train"], split["test"]
    else:
        return full, None


def make_preprocess_fn(tokenizer, max_prompt_len: int, max_target_len: int):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    def preprocess(batch):
        input_ids_batch =[]
        attention_mask_batch = []
        labels_batch = []
        for prompt, target in zip(batch["prompt"], batch["target"]):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]
                
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if len(target_ids) > max_target_len - 1:
                target_ids = target_ids[: (max_target_len - 1)]
                
            target_ids = target_ids + [eos_id]
            full_ids = prompt_ids + target_ids
            
            attention_mask = [1] * len(full_ids)
            labels = [-100] * len(prompt_ids) + target_ids
            
            max_total = max_prompt_len + max_target_len
            pad_length = max_total - len(full_ids)
            if pad_length > 0:
                full_ids = full_ids +[pad_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [-100] * pad_length
            else:
                full_ids = full_ids[:max_total]
                attention_mask = attention_mask[:max_total]
                labels = labels[:max_total]
                
            input_ids_batch.append(full_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)
        return {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "labels": labels_batch}

    return preprocess


def numpy_collate(features: List[Dict]):
    import numpy as np
    input_ids = np.array([f["input_ids"] for f in features], dtype=np.int64)
    attention_mask = np.array([f["attention_mask"] for f in features], dtype=np.int64)
    labels = np.array([f["labels"] for f in features], dtype=np.int64)
    
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }
    return batch


def main():
    print("Started training script.")
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B with QLoRA + LoRA")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, default="./finetuned_model", help="Output directory")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--save-hours", type=float, default=2.0)
    parser.add_argument("--estimate-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=None, help="Explicit save_steps")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, val_ds = load_dataset(args.data, val_split=args.val_split, seed=args.seed)
    print(f"Loaded dataset. Train: {len(train_ds)}; Val: {len(val_ds) if val_ds is not None else 0}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print(
        f"Tokenizer IDs → BOS:{tokenizer.bos_token_id} "
        f"EOS:{tokenizer.eos_token_id} "
        f"PAD:{tokenizer.pad_token_id}")

    preprocess_fn = make_preprocess_fn(tokenizer, args.max_prompt_len, args.max_target_len)
    print("Tokenizing train dataset...")
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    if val_ds is not None:
        print("Tokenizing val dataset...")
        val_ds = val_ds.map(preprocess_fn, batched=True, remove_columns=val_ds.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model ({args.model}) with 4-bit QLoRA config ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied. Trainable: {trainable:,} / Total: {total:,} ({100.0 * trainable / total:.3f}%)")

    # FIX 1: Manually calculate `warmup_steps` instead of using the deprecated `warmup_ratio`
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_train_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_train_steps * 0.05)) # 5% of total steps
    
    logging_steps = 50
    save_steps = args.save_steps if args.save_steps is not None else 500

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,  # Fixed deprecation warning here
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_ds is not None else "no",
        gradient_checkpointing=True,
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=True if val_ds is not None else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # FIX 2: Replaced `tokenizer` with `processing_class`
        data_collator=numpy_collate,
    )

    if args.estimate_steps and args.estimate_steps > 0:
        print(f"Running short estimate run for {args.estimate_steps} training steps...")
        t0 = time.time()
        tmp_save_steps = trainer.args.save_steps
        tmp_eval_steps = trainer.args.eval_steps
        tmp_max_steps = trainer.args.max_steps
        
        trainer.args.save_steps = 999999999
        trainer.args.eval_steps = 999999999
        trainer.args.logging_steps = logging_steps
        trainer.args.max_steps = args.estimate_steps # FIX 3: Moved max_steps out of trainer.train()

        trainer.train(resume_from_checkpoint=None)
        
        t1 = time.time()
        elapsed = t1 - t0
        secs_per_step = elapsed / args.estimate_steps
        steps_per_hour = 3600.0 / secs_per_step
        suggested_save_steps = max(1, int(steps_per_hour * args.save_hours))
        print(f"Estimate complete: {secs_per_step:.3f} sec/step, ~{steps_per_hour:.1f} steps/hour")
        print(f"Suggested save_steps for checkpoint every {args.save_hours} hours: {suggested_save_steps}")
        
        trainer.args.save_steps = tmp_save_steps
        trainer.args.eval_steps = tmp_eval_steps
        trainer.args.max_steps = tmp_max_steps
        
        print("Saving intermediate model from estimate run...")
        trainer.save_model(os.path.join(args.output, "estimate_checkpoint"))
        tokenizer.save_pretrained(os.path.join(args.output, "estimate_checkpoint"))
        print("Exiting after estimate run as requested.")
        return

    print("Starting full training run")
    trainer.train()
    print("Training finished. Saving model and tokenizer...")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved adapters & tokenizer to: {args.output}")

if __name__ == "__main__":
    main()