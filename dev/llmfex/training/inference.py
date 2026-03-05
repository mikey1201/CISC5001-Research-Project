"""
Inference Script for Fine-tuned PDE Operator Prediction Model

This script loads a fine-tuned Llama 3.2 3B model and uses it to predict
the operator sequence for PDE solutions.

Usage:
    python inference.py --model ./finetuned_model --prompt "Type: Poisson | RHS: ..."

Or with interactive mode:
    python inference.py --model ./finetuned_model --interactive
"""

import os
import json
import argparse
import torch
from typing import List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel


def load_finetuned_model(
    model_path: str,
    base_model_name: str = "meta-llama/Llama-3.2-3B",
    device: str = "cuda",
):
    """
    Load the fine-tuned model with LoRA adapters.
    
    Args:
        model_path: Path to the fine-tuned model (LoRA adapters)
        base_model_name: Name of the base model
        device: Device to load the model on
        
    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if os.path.exists(os.path.join(model_path, "tokenizer.json")) 
        else base_model_name,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def predict_operators(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
) -> str:
    """
    Generate operator sequence prediction for a PDE input.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt: Input prompt (PDE specification)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Predicted operator sequence string
    """
    # Ensure prompt ends with solution marker
    if not prompt.endswith("||"):
        if "Solution:" in prompt:
            if not prompt.rstrip().endswith("||"):
                prompt = prompt.rstrip() + " ||"
        else:
            prompt = prompt + " | Solution: ||"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Prevent repetition
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the predicted part (after the prompt)
    if "||" in full_output:
        # Find the last occurrence of || (the one we added)
        parts = full_output.split("||")
        if len(parts) > 1:
            prediction = parts[-1].strip()
        else:
            prediction = full_output
    else:
        prediction = full_output
    
    return prediction


def extract_operators(prediction: str) -> List[str]:
    """
    Extract unique operators from the predicted sequence.
    
    Paper Section 3.4:
    "The first step involves identifying the unique set of operators 
    by removing duplicate tokens from the sequence."
    """
    # Valid operators
    valid_ops = {
        # Binary
        '+', '-', '*', '/',
        # Unary
        'sin', 'cos', 'tan', 'exp', 'ln', 'sqrt', 'abs',
        # Powers
        '^2', '^3', '^4', '^',
        # Variables
        'x0', 'x1', 'x2', 'x3', 'x4',
        # Constants
        'const',
    }
    
    tokens = prediction.split()
    operators = []
    seen = set()
    
    for token in tokens:
        token = token.strip()
        if token in valid_ops and token not in seen:
            operators.append(token)
            seen.add(token)
    
    return operators


def compute_mismatch(
    predicted_ops: List[str],
    ground_truth_ops: List[str],
) -> int:
    """
    Compute the number of mismatched operators.
    
    Paper Section 3.4 and Table 1:
    Uses binary vector encoding and computes ||y - z||^2
    which equals the number of mismatched operators.
    """
    pred_set = set(predicted_ops)
    true_set = set(ground_truth_ops)
    
    # Symmetric difference gives mismatched operators
    mismatch = pred_set.symmetric_difference(true_set)
    return len(mismatch)


def interactive_mode(model, tokenizer):
    """
    Run interactive inference session.
    """
    print("\n" + "="*60)
    print("Interactive PDE Operator Prediction")
    print("="*60)
    print("Enter PDE specifications to get operator predictions.")
    print("Type 'quit' to exit.")
    print("="*60 + "\n")
    
    while True:
        print("\nEnter prompt (or 'quit'):")
        prompt = input("> ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        try:
            prediction = predict_operators(model, tokenizer, prompt)
            operators = extract_operators(prediction)
            
            print(f"\nPredicted sequence: {prediction}")
            print(f"Extracted operators: {operators}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="PDE Operator Prediction Inference")
    parser.add_argument("--model", type=str, default="./finetuned_model",
                        help="Path to fine-tuned model")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Base model name")
    parser.add_argument("--prompt", type=str, default=None,
                        help="PDE specification prompt")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test data JSON for evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_finetuned_model(args.model, args.base_model)
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    
    elif args.test_data:
        # Evaluate on test data
        print(f"\nEvaluating on: {args.test_data}")
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        
        total_mismatch = 0
        num_samples = min(100, len(test_data))  # Evaluate on first 100
        
        for i, sample in enumerate(test_data[:num_samples]):
            prompt = sample['input']
            ground_truth = sample['target']
            
            prediction = predict_operators(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            
            pred_ops = extract_operators(prediction)
            true_ops = extract_operators(ground_truth)
            mismatch = compute_mismatch(pred_ops, true_ops)
            total_mismatch += mismatch
            
            if i < 5:  # Print first 5 examples
                print(f"\n--- Sample {i+1} ---")
                print(f"Input: {prompt[:80]}...")
                print(f"True: {ground_truth}")
                print(f"Pred: {prediction}")
                print(f"Mismatch: {mismatch}")
        
        avg_mismatch = total_mismatch / num_samples
        print(f"\n{'='*60}")
        print(f"Average mismatch: {avg_mismatch:.4f}")
        print(f"Total samples: {num_samples}")
        print(f"{'='*60}")
    
    elif args.prompt:
        # Single prediction
        prediction = predict_operators(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        operators = extract_operators(prediction)
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Prediction: {prediction}")
        print(f"Extracted operators: {operators}")
    
    else:
        # Default: run interactive mode
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
