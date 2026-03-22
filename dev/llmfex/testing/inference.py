#!/usr/bin/env python3
"""
inference.py

Inference script for evaluating fine-tuned LLMs on PDE operator prediction.
Follows the methodology from "From Equations to Insights: Unraveling Symbolic 
Structures in PDEs with LLMs" (Bhatnagar et al.)

Key components:
1. Model loading with LoRA adapters (QLoRA inference)
2. Autoregressive generation of operator sequences
3. Post-processing to extract unique operator sets
4. Binary vector encoding for operator comparison
5. Squared ℓ₂-norm mismatch calculation (as defined in the paper)
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================================
# OPERATOR DICTIONARY (as used in the paper)
# ============================================================================

# Fixed operator dictionary for binary vector encoding
# This includes all operators that can appear in PDE solution expressions
OPERATOR_DICTIONARY = [
    # Variables
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
    # Constants (represented as 'const' in postfix notation)
    "const",
    # Binary operators
    "+", "-", "*", "/",
    # Power operators (special notation used in paper)
    "^2", "^3", "^4", "^5", "^6",
    # Unary operators (trigonometric)
    "sin", "cos", "tan",
    # Unary operators (exponential/logarithmic)
    "exp", "ln", "lg",
    # Other unary operators
    "sqrt", "abs",
    # Generic power (for arbitrary exponents)
    "pow",
]

# Create a mapping from operator name to index
OPERATOR_TO_IDX = {op: idx for idx, op in enumerate(OPERATOR_DICTIONARY)}
NUM_OPERATORS = len(OPERATOR_DICTIONARY)


# ============================================================================
# POSTFIX NOTATION UTILITIES
# ============================================================================

def extract_operators_from_postfix(postfix_str: str) -> Set[str]:
    """
    Extract the unique set of operators from a postfix (RPN) expression.
    
    This follows the paper's approach of post-processing the model output
    to extract a clean and interpretable set of predicted operators.
    
    Args:
        postfix_str: A space-separated postfix notation string
        
    Returns:
        Set of unique operators found in the expression
    """
    if not postfix_str or not postfix_str.strip():
        return set()
    
    tokens = postfix_str.strip().split()
    operators = set()
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
            
        # Check if token matches known operators
        if token in OPERATOR_TO_IDX:
            operators.add(token)
        else:
            # Handle numeric constants
            try:
                float(token)
                operators.add("const")
            except ValueError:
                # Try to match patterns like ^2, ^3, etc.
                if token.startswith("^") and len(token) > 1:
                    try:
                        exp = int(token[1:])
                        if 2 <= exp <= 6:
                            operators.add(token)
                        else:
                            operators.add("pow")
                    except ValueError:
                        operators.add("pow")
                # Unknown token - could be a variable with different naming
                elif token.startswith("x") and token[1:].isdigit():
                    operators.add(token)
                else:
                    # Try lowercase version
                    token_lower = token.lower()
                    if token_lower in OPERATOR_TO_IDX:
                        operators.add(token_lower)
    
    return operators


def validate_operators(operators: Set[str]) -> Set[str]:
    """
    Validate and filter operators, removing misspelled or malformed tokens.
    
    This implements the error-checking step mentioned in Section 3.4 of the paper:
    "Such errors are particularly likely when dealing with rare or domain-specific 
    mathematical symbols. Any invalid or unrecognized tokens are subsequently 
    discarded to ensure the integrity of the final operator set."
    
    Args:
        operators: Set of extracted operators
        
    Returns:
        Set of valid operators
    """
    valid_operators = set()
    for op in operators:
        if op in OPERATOR_TO_IDX:
            valid_operators.add(op)
        elif op.startswith("x") and op[1:].isdigit():
            # Valid variable name
            valid_operators.add(op)
    return valid_operators


# ============================================================================
# BINARY VECTOR ENCODING (Paper's Methodology - Section 3.4)
# ============================================================================

def operators_to_binary_vector(operators: Set[str]) -> torch.Tensor:
    """
    Encode an operator set as a binary vector over a fixed dictionary.
    
    This follows the paper's exact methodology from Section 3.4:
    "Specifically, we encode each operator set as a binary vector over a 
    fixed dictionary of n possible operators, where each vector component 
    indicates the presence or absence of a given operator."
    
    Args:
        operators: Set of operator names
        
    Returns:
        Binary tensor of shape (NUM_OPERATORS,) where 1 indicates presence
    """
    binary_vector = torch.zeros(NUM_OPERATORS, dtype=torch.float32)
    
    for op in operators:
        if op in OPERATOR_TO_IDX:
            binary_vector[OPERATOR_TO_IDX[op]] = 1.0
        elif op.startswith("x") and op[1:].isdigit():
            # Handle variables beyond x10 by mapping to nearest index
            var_num = int(op[1:])
            if var_num <= 10:
                binary_vector[OPERATOR_TO_IDX[f"x{var_num}"]] = 1.0
            else:
                # Map to x10 slot for variables beyond our dictionary
                binary_vector[OPERATOR_TO_IDX["x10"]] = 1.0
    
    return binary_vector


def compute_mismatch(predicted_ops: Set[str], ground_truth_ops: Set[str]) -> int:
    """
    Compute the squared ℓ₂-norm of the difference between predicted and 
    ground-truth operator sets.
    
    This follows the paper's exact methodology from Section 3.4:
    "Then, the squared distance between two such operator sets, represented by 
    two binary vectors y ∈ Rⁿ and z ∈ Rⁿ, can be defined as 
    ||y - z||² = Σᵢ(yᵢ - zᵢ)². Clearly, it measures the number of mismatched 
    operators between the two operator sets."
    
    Args:
        predicted_ops: Set of predicted operators
        ground_truth_ops: Set of ground truth operators
        
    Returns:
        Number of mismatched operators (integer)
    """
    pred_vector = operators_to_binary_vector(predicted_ops)
    gt_vector = operators_to_binary_vector(ground_truth_ops)
    
    # Squared ℓ₂-norm: ||y - z||² = Σᵢ(yᵢ - zᵢ)²
    difference = pred_vector - gt_vector
    squared_l2_norm = torch.sum(difference ** 2).item()
    
    return int(squared_l2_norm)


# ============================================================================
# MODEL LOADING AND INFERENCE
# ============================================================================

def load_model(
    model_path: str,
    base_model_name: str = "meta-llama/Llama-3.2-3B",
    device: str = "auto",
    load_in_4bit: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the fine-tuned model with LoRA adapters.
    
    Args:
        model_path: Path to the LoRA adapter weights
        base_model_name: Name or path of the base model
        device: Device mapping strategy
        load_in_4bit: Whether to load in 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading base model {base_model_name}...")
    
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    print(f"Loading LoRA adapters from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_prediction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> str:
    """
    Generate a prediction for a given prompt using autoregressive decoding.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 for greedy)
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated text (decoded tokens)
    """
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False,
    )
    
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text.strip()


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def parse_prompt_and_target(data_item: Dict) -> Tuple[str, str]:
    """
    Parse a data item to extract the prompt and ground truth target.
    
    Args:
        data_item: Dictionary with 'prompt' and 'target' fields
        
    Returns:
        Tuple of (prompt, target)
    """
    prompt = data_item.get("prompt", "")
    target = data_item.get("target", "")
    
    # If prompt doesn't end with "Solution: ", add it
    if not prompt.endswith("Solution: "):
        if "Solution:" in prompt:
            # Extract up to and including "Solution: "
            idx = prompt.rfind("Solution:")
            prompt = prompt[:idx + len("Solution:")]
        else:
            prompt = prompt + " Solution: "
    
    # Ensure prompt ends with "Solution: "
    if not prompt.rstrip().endswith("Solution:"):
        prompt = prompt.rstrip() + " Solution: "
    
    return prompt, target


def evaluate_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[Dict],
    max_new_tokens: int = 64,
    verbose: bool = False,
    show_predictions: bool = False,
    max_display: int = 50,
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset and compute average mismatch.
    
    This implements the evaluation methodology from Section 5.2 of the paper:
    "In this subsection, we evaluate the performance of the fine-tuned BART, 
    T5 and Llama3 models in predicting operator sets. Specifically, we track 
    the average number of mismatched operators on the test dataset for each epoch."
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        data: List of data items with 'prompt' and 'target' fields
        max_new_tokens: Maximum tokens to generate
        verbose: Whether to print progress
        show_predictions: Whether to print prediction vs ground truth for each sample
        max_display: Maximum number of samples to display when show_predictions is True
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_mismatch = 0
    total_samples = 0
    exact_matches = 0
    mismatches = []
    predictions_detail = []
    
    for i, item in enumerate(data):
        prompt, target = parse_prompt_and_target(item)
        
        # Generate prediction
        prediction = generate_prediction(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        
        # Extract operators
        pred_ops = extract_operators_from_postfix(prediction)
        pred_ops = validate_operators(pred_ops)
        
        gt_ops = extract_operators_from_postfix(target)
        gt_ops = validate_operators(gt_ops)
        
        # Compute mismatch
        mismatch = compute_mismatch(pred_ops, gt_ops)
        mismatches.append(mismatch)
        total_mismatch += mismatch
        total_samples += 1
        
        if mismatch == 0:
            exact_matches += 1
        
        # Store prediction details
        sample_detail = {
            "index": i,
            "prompt": prompt,
            "prediction": prediction,
            "ground_truth": target,
            "predicted_operators": sorted(pred_ops),
            "ground_truth_operators": sorted(gt_ops),
            "mismatch": mismatch,
        }
        predictions_detail.append(sample_detail)
        
        # Show individual predictions if requested
        if show_predictions and i < max_display:
            print(f"\n{'='*70}")
            print(f"Sample {i + 1}")
            print(f"{'='*70}")
            print(f"PROMPT: {prompt[:100]}..." if len(prompt) > 100 else f"PROMPT: {prompt}")
            print(f"\nPREDICTION:      {prediction}")
            print(f"GROUND TRUTH:    {target}")
            print(f"\nPREDICTED OPS:   {sorted(pred_ops)}")
            print(f"GROUND TRUTH OPS:{sorted(gt_ops)}")
            print(f"\nMISMATCH: {mismatch}")
            if mismatch == 0:
                print("✓ EXACT MATCH!")
            else:
                # Show which operators are different
                missing = gt_ops - pred_ops
                extra = pred_ops - gt_ops
                if missing:
                    print(f"  Missing operators: {sorted(missing)}")
                if extra:
                    print(f"  Extra operators:   {sorted(extra)}")
        
        if verbose and (i + 1) % 100 == 0:
            avg_mismatch = total_mismatch / total_samples
            print(f"\n[Progress] Processed {i + 1}/{len(data)} samples. Avg mismatch: {avg_mismatch:.4f}")
    
    avg_mismatch = total_mismatch / total_samples if total_samples > 0 else 0
    exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0
    
    if show_predictions and len(data) > max_display:
        print(f"\n[Note] Displayed {max_display} of {len(data)} samples. Use --max_display to see more.")
    
    results = {
        "average_mismatch": avg_mismatch,
        "total_samples": total_samples,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "mismatches": mismatches,
        "predictions_detail": predictions_detail,
    }
    
    return results


# ============================================================================
# BATCH INFERENCE
# ============================================================================

def batch_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 64,
) -> List[str]:
    """
    Run batch inference on multiple prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        batch_size: Batch size for inference
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        List of generated predictions
    """
    predictions = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode outputs
        for j, output in enumerate(outputs):
            generated_ids = output[input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predictions.append(generated_text.strip())
    
    return predictions


# ============================================================================
# MAIN INFERENCE SCRIPT
# ============================================================================

def read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file and return a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for PDE operator prediction"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model (LoRA adapters)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to evaluation data (JSONL format)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for interactive inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--show_predictions",
        action="store_true",
        help="Show prediction vs ground truth for each sample",
    )
    parser.add_argument(
        "--max_display",
        type=int,
        default=50,
        help="Maximum number of samples to display with --show_predictions (default: 50)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save detailed predictions to output file (in addition to metrics)",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        base_model_name=args.base_model,
        load_in_4bit=not args.no_4bit,
    )
    
    # Single prompt inference
    if args.prompt:
        print("\n" + "=" * 60)
        print("PROMPT:")
        print(args.prompt)
        print("=" * 60)
        
        prediction = generate_prediction(
            model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens
        )
        
        print("\nPREDICTION:")
        print(prediction)
        print("=" * 60)
        
        operators = extract_operators_from_postfix(prediction)
        operators = validate_operators(operators)
        print("\nEXTRACTED OPERATORS:")
        print(sorted(operators))
        print("=" * 60)
        return
    
    # Dataset evaluation
    if args.data:
        print(f"\nLoading data from {args.data}...")
        data = read_jsonl(args.data)
        print(f"Loaded {len(data)} samples")
        
        print("\nRunning evaluation...")
        results = evaluate_dataset(
            model, tokenizer, data,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
            show_predictions=args.show_predictions,
            max_display=args.max_display,
        )
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS:")
        print(f"  Total samples:    {results['total_samples']}")
        print(f"  Average mismatch: {results['average_mismatch']:.4f}")
        print(f"  Exact matches:    {results['exact_matches']}")
        print(f"  Exact match rate: {results['exact_match_rate']:.4f}")
        print("=" * 60)
        
        # Save results
        output_data = {
            "model_path": args.model_path,
            "data_path": args.data,
            "metrics": {
                "average_mismatch": results["average_mismatch"],
                "total_samples": results["total_samples"],
                "exact_matches": results["exact_matches"],
                "exact_match_rate": results["exact_match_rate"],
            },
            "mismatches": results["mismatches"],
        }
        
        # Optionally include detailed predictions
        if args.save_predictions:
            output_data["predictions"] = results["predictions_detail"]
            print(f"  (Including {len(results['predictions_detail'])} detailed predictions)")
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")
        return
    
    print("Please provide either --data or --prompt argument")


if __name__ == "__main__":
    main()
