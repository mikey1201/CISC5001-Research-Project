"""
LLM Fine-tuning for PDE Operator Prediction

This script fine-tunes Llama 3.2 3B (base model) to predict operator
sequences in PDE solutions, following the paper:

"From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs"
by Bhatnagar, Liang, Patel, and Yang

Key features:
- Works with the new data pipeline format (input/target fields)
- LoRA for parameter-efficient fine-tuning
- 4-bit quantization for memory efficiency
- Optimized for RTX 5070 Ti (16GB VRAM)
- Decoder-only model architecture (Llama family)

Data Format Expected:
{
    "input": "Type: Poisson | RHS: const x0 * | Cauchy: x2=0.0 const ||| const | Solution: ||",
    "target": "const x0 ^3 + const x1 x2 * cos * +",
    "full": "Type: Poisson | RHS: ... | Solution: || const x0 ^3 + ...",
    ...
}
"""

import os
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import torch
import numpy as np
from tqdm import tqdm

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset
import yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model and training - optimized for RTX 5070 Ti"""
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.2-3B"
    max_prompt_length: int = 256
    max_target_length: int = 64
    total_sequence_length: int = 320  # prompt + target + buffer
    
    # LoRA settings (from paper Section 5.1)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Target modules for Llama architecture (attention layers)
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training settings (adapted from paper for single GPU)
    num_epochs: int = 3
    batch_size: int = 4  # Conservative for 16GB VRAM
    gradient_accumulation_steps: int = 8  # Effective batch = 4 × 8 = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    
    # Precision and memory
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True  # Enable for memory savings
    max_grad_norm: float = 1.0
    
    # Paths
    data_path: str = "./data/pde_dataset.json"
    output_dir: str = "./finetuned_model"
    run_name: str = "llama_fex_pde"
    
    # Logging
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Validation split
    val_split: float = 0.05  # 5% for validation


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_dataset(data_path: str, val_split: float = 0.05) -> tuple:
    """
    Load the JSON dataset and split into train/validation.
    
    Args:
        data_path: Path to the JSON file
        val_split: Fraction of data to use for validation
        
    Returns:
        (train_dataset, val_dataset) as Hugging Face Dataset objects
    """
    print(f"Loading dataset from: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Validate data format
    required_fields = ['input', 'target']
    for i, sample in enumerate(data[:5]):
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Sample {i} missing required field: {field}")
    
    # Print sample for verification
    print("\n" + "="*60)
    print("Sample data point:")
    print(f"  Input: {data[0]['input'][:100]}...")
    print(f"  Target: {data[0]['target'][:50]}...")
    print("="*60 + "\n")
    
    # Convert to Dataset
    full_dataset = Dataset.from_list(data)
    
    # Split into train/validation
    if val_split > 0:
        split = full_dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split['train']
        val_dataset = split['test']
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"Train samples: {len(train_dataset)}")
        print(f"No validation split")
    
    return train_dataset, val_dataset


def preprocess_decoder_only(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
    max_target_length: int,
) -> Dict:
    """
    Preprocess examples for decoder-only model training.
    
    For causal language modeling, we concatenate prompt and target,
    then mask the prompt portion in the loss calculation.
    
    Input format from data pipeline:
        input: "Type: Poisson | RHS: ... | Solution: ||"
        target: "const x0 ^3 + ..."
    
    The model should predict the target given the input.
    """
    
    prompts = examples['input']
    targets = examples['target']
    
    max_length = max_prompt_length + max_target_length
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for prompt, target in zip(prompts, targets):
        # Tokenize prompt (with special tokens)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        
        # Truncate prompt if too long
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]
        
        # Tokenize target (without special tokens - it's a continuation)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        
        # Truncate target if too long
        if len(target_ids) > max_target_length:
            target_ids = target_ids[:max_target_length]
        
        # Add EOS token at the end
        target_ids = target_ids + [tokenizer.eos_token_id]
        
        # Concatenate prompt and target
        full_ids = prompt_ids + target_ids
        
        # Create attention mask (1 for real tokens)
        attention_mask = [1] * len(full_ids)
        
        # Create labels: mask prompt with -100, keep target
        labels = [-100] * len(prompt_ids) + target_ids
        
        # Pad to max_length
        padding_length = max_length - len(full_ids)
        if padding_length > 0:
            full_ids = full_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
        elif padding_length < 0:
            # Truncate if still too long
            full_ids = full_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        
        input_ids_list.append(full_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list,
    }


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model_and_tokenizer(config: TrainingConfig):
    """
    Load the base model and tokenizer with 4-bit quantization.
    
    Uses NF4 quantization for optimal memory efficiency on consumer GPUs.
    """
    
    print(f"\n{'='*60}")
    print("Loading Model and Tokenizer")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Precision: {'BF16' if config.bf16 else 'FP16' if config.fp16 else 'FP32'}")
    print(f"Quantization: 4-bit NF4")
    print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    print(f"{'='*60}\n")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for additional savings
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Ensure pad token is set (Llama doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        attn_implementation="eager",  # Avoid flash attention compatibility issues
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Total parameters: {total_params / 1e9:.2f}B")
    
    return model, tokenizer


def setup_lora(model, config: TrainingConfig):
    """
    Configure LoRA for parameter-efficient fine-tuning.
    
    Paper Section 5.1 settings:
    - Rank: 16
    - Alpha: 32 (scaling factor)
    - Dropout: 0.1
    - Target modules: attention layers (q_proj, v_proj, k_proj, o_proj)
    """
    
    print(f"\n{'='*60}")
    print("Setting up LoRA")
    print(f"{'='*60}")
    print(f"Rank: {config.lora_rank}")
    print(f"Alpha: {config.lora_alpha}")
    print(f"Dropout: {config.lora_dropout}")
    print(f"Target modules: {config.target_modules}")
    print(f"{'='*60}\n")
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",  # No bias adaptation
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: TrainingConfig,
):
    """
    Train the model using Hugging Face Trainer.
    """
    
    print(f"\n{'='*60}")
    print("Preparing Training")
    print(f"{'='*60}")
    
    # Preprocess datasets
    print("Tokenizing training dataset...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_decoder_only(
            x, tokenizer, config.max_prompt_length, config.max_target_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    
    if val_dataset is not None:
        print("Tokenizing validation dataset...")
        val_dataset = val_dataset.map(
            lambda x: preprocess_decoder_only(
                x, tokenizer, config.max_prompt_length, config.max_target_length
            ),
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val",
        )
    
    # Calculate training steps
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch_size
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size per device: {config.batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"{'='*60}\n")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        
        # Training hyperparameters
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        
        # Precision
        bf16=config.bf16,
        fp16=config.fp16,
        
        # Memory optimization
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Optimizer and scheduler
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        
        # Logging
        logging_steps=config.logging_steps,
        logging_dir=os.path.join(config.output_dir, "logs"),
        
        # Evaluation and saving
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True if val_dataset else False,
        
        # Other
        report_to="none",  # Set to "wandb" for experiment tracking
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Suppress unnecessary warnings
        logging_nan_inf_filter=False,
    )
    
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # Save final model
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Saving model to: {config.output_dir}")
    
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training config
    config_save_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    print(f"Model saved successfully!")
    print(f"{'='*60}\n")
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B for PDE operator prediction")
    
    # Data arguments
    parser.add_argument("--data", type=str, default="./data/pde_dataset.json",
                        help="Path to training data JSON file")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Validation split fraction (default: 0.05)")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Model name or path (default: meta-llama/Llama-3.2-3B)")
    
    # Training arguments
    parser.add_argument("--output", type=str, default="./finetuned_model",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    
    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    
    # Memory arguments
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing for memory savings")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", 
                        dest="gradient_checkpointing",
                        help="Disable gradient checkpointing")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides other arguments)")
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("LLM Fine-tuning for PDE Operator Prediction")
    print("Based on: 'From Equations to Insights: Unraveling Symbolic")
    print("          Structures in PDEs with LLMs'")
    print("="*60 + "\n")
    
    # Create config
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # TODO: Parse YAML config
        config = TrainingConfig()
    else:
        config = TrainingConfig(
            model_name=args.model,
            data_path=args.data,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            gradient_checkpointing=args.gradient_checkpointing,
            val_split=args.val_split,
        )
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    train_dataset, val_dataset = load_dataset(config.data_path, config.val_split)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Train
    train_model(model, tokenizer, train_dataset, val_dataset, config)
    
    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {config.output_dir}")
    print("="*60 + "\n")
    
    # Print usage example
    print("To use the fine-tuned model for inference:")
    print(f"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained("{config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained("{config.output_dir}")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, "{config.output_dir}")
    
    # Inference
    prompt = "Type: Poisson | RHS: const x0 * | Dirichlet: x0=0 const | Solution: ||"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs[0]))
    """)


if __name__ == "__main__":
    main()
