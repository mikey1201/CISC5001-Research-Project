#!/usr/bin/env python3
"""
Bridge: LLM Inference to FEX Grammar

This script connects your fine-tuned LLM model to the FEX grammar system.
It takes a PDE problem description, runs inference with your model,
extracts operators from the predicted postfix sequence, and generates
a FEX-compatible grammar file.

Usage:
    python llm_to_fex_bridge.py --model_path ./your_finetuned_model \
        --pde_type Poisson \
        --rhs "const" \
        --boundary "Dirichlet: x1=0 const" \
        --output grammar.json

Then run FEX with:
    python controller_poisson_grammar.py --grammar_source llm --grammar_path grammar.json
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Set, Optional, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Finite-expression-method', 'fex'))

# ============================================================================
# OPERATOR MAPPING: LLM tokens → FEX operators
# ============================================================================

# Your LLM's operator dictionary (from inference.py)
LLM_OPERATOR_DICTIONARY = [
    # Variables (mapped to identity in FEX)
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
    # Constants
    "const",
    # Binary operators
    "+", "-", "*", "/",
    # Power operators
    "^2", "^3", "^4", "^5", "^6",
    # Unary operators (trigonometric)
    "sin", "cos", "tan",
    # Unary operators (exponential/logarithmic)
    "exp", "ln", "lg",
    # Other unary operators
    "sqrt", "abs",
    # Generic power
    "pow",
]

# Mapping from LLM tokens to FEX operator names
LLM_TO_FEX_MAPPING = {
    # Binary operators
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    
    # Unary operators
    "sin": "sin",
    "cos": "cos",
    "tan": "tanh",  # FEX doesn't have tan, use tanh
    "exp": "exp",
    "ln": "log",
    "lg": "log",    # Map log base 10 to natural log
    "sqrt": "sqrt",
    "abs": "abs",
    
    # Power operators (each power is a SEPARATE unary operator)
    "^2": "square",
    "^3": "cube",
    "^4": "quad",
    "^5": "pow5",
    "^6": "pow6",
    "^7": "pow7",   # NEW: Added power 7
    "^8": "pow8",   # NEW: Added power 8
    "^9": "pow9",   # NEW: Added power 9
    "^10": "pow10", # NEW: Added power 10
    "pow": "square", # Generic pow defaults to square (most common)
    
    # Variables (identity function in FEX)
    "x1": "identity",
    "x2": "identity",
    "x3": "identity",
    "x4": "identity",
    "x5": "identity",
    "x6": "identity",
    "x7": "identity",
    "x8": "identity",
    "x9": "identity",
    "x10": "identity",
    
    # Constants
    "const": "one",  # Map const to one (constant function)
}

# Operators to include in FEX grammar (excludes variables and const)
FEX_OPERATOR_TYPES = {
    "add": "binary",
    "sub": "binary",
    "mul": "binary",
    "div": "binary",
    "sin": "unary",
    "cos": "unary",
    "tanh": "unary",
    "exp": "unary",
    "log": "unary",
    "sqrt": "unary",
    "abs": "unary",
    "square": "unary",
    "cube": "unary",
    "quad": "unary",
    "pow5": "unary",
    "pow6": "unary",
    "pow7": "unary",
    "pow8": "unary",
    "pow9": "unary",
    "pow10": "unary",
    "identity": "unary",
    "one": "unary",
}


# ============================================================================
# POSTFIX PARSING (from your inference.py)
# ============================================================================

def extract_operators_from_postfix(postfix_str: str) -> Set[str]:
    """
    Extract unique operators from a postfix (RPN) expression.
    
    Args:
        postfix_str: Space-separated postfix notation string
        
    Returns:
        Set of unique operator tokens
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
        if token in LLM_OPERATOR_DICTIONARY:
            operators.add(token)
        else:
            # Try lowercase
            token_lower = token.lower()
            if token_lower in LLM_OPERATOR_DICTIONARY:
                operators.add(token_lower)
            # Handle numeric constants
            else:
                try:
                    float(token)
                    operators.add("const")
                except ValueError:
                    # Check for power notation
                    if token.startswith("^") and len(token) > 1:
                        operators.add(token)
    
    return operators


def validate_operators(operators: Set[str]) -> Set[str]:
    """Remove invalid/unrecognized tokens."""
    valid = set()
    for op in operators:
        if op in LLM_OPERATOR_DICTIONARY:
            valid.add(op)
        elif op.startswith("^"):
            # Keep power operators
            valid.add(op)
    return valid


def llm_operators_to_fex_grammar(llm_operators: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Convert LLM operator tokens to FEX operator names.
    
    Args:
        llm_operators: Set of operators from LLM output
        
    Returns:
        Tuple of (unary_operators, binary_operators) for FEX
    """
    unary = set()
    binary = set()
    
    for llm_op in llm_operators:
        # Map to FEX name
        fex_name = LLM_TO_FEX_MAPPING.get(llm_op, llm_op)
        
        # Check if it's a valid FEX operator
        if fex_name in FEX_OPERATOR_TYPES:
            op_type = FEX_OPERATOR_TYPES[fex_name]
            if op_type == "unary":
                unary.add(fex_name)
            else:
                binary.add(fex_name)
        else:
            print(f"Warning: LLM operator '{llm_op}' not supported in FEX, skipping")
    
    return sorted(unary), sorted(binary)


# ============================================================================
# GRAMMAR FILE GENERATION
# ============================================================================

def generate_fex_grammar_file(
    unary_operators: List[str],
    binary_operators: List[str],
    output_path: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Generate a FEX-compatible grammar JSON file.
    
    Args:
        unary_operators: List of unary operator names
        binary_operators: List of binary operator names
        output_path: Path to save the grammar file
        metadata: Optional metadata to include
        
    Returns:
        The grammar dictionary
    """
    grammar = {
        "name": "LLM_Generated_Grammar",
        "grammar_type": "llm_generated",
        "version": "1.0",
        "unary_operators": [{"name": op} for op in unary_operators],
        "binary_operators": [{"name": op} for op in binary_operators],
        "metadata": metadata or {
            "source": "llm_inference",
            "description": "Grammar generated from LLM-predicted operator sequence"
        }
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(grammar, f, indent=2)
    
    print(f"Grammar saved to: {output_path}")
    return grammar


# ============================================================================
# LLM INFERENCE (optional - requires model)
# ============================================================================

def run_llm_inference(
    model_path: str,
    prompt: str,
    base_model: str = "meta-llama/Llama-3.2-3B",
    max_new_tokens: int = 64,
    use_quantization: bool = False,
    device: str = "auto",
) -> str:
    """
    Run inference with your fine-tuned LLM model.
    
    This is optional - if you don't have torch/transformers installed,
    you can manually provide the postfix sequence.
    
    Args:
        model_path: Path to your fine-tuned model (LoRA adapters)
        prompt: Input prompt (PDE description)
        base_model: Base model name
        max_new_tokens: Maximum tokens to generate
        use_quantization: Whether to use 4-bit quantization (requires CUDA 13.x)
        device: Device to use ("auto", "cuda", "cpu", "mps")
        
    Returns:
        Generated postfix sequence
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "transformers, torch, and peft are required for LLM inference. "
            "Install with: pip install transformers torch peft"
        )
    
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine the best available device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Apple Silicon) device")
        else:
            device = "cpu"
            print("Using CPU device (slow - consider using a GPU)")
    
    # Determine torch dtype based on device
    if device == "cuda" and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    print(f"Using dtype: {torch_dtype}")
    
    # Try loading with quantization first if requested and available
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            
            print("Attempting 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Falling back to full precision loading...")
            use_quantization = False
    
    # Load without quantization
    if not use_quantization:
        print(f"Loading base model {base_model} in {torch_dtype}...")
        
        # For CPU, we need to handle memory carefully
        if device == "cpu":
            # Try low_cpu_mem_usage for large models
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                print(f"low_cpu_mem_usage failed, trying standard loading: {e}")
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
    
    # Load LoRA adapters
    print("Loading LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(base, model_path)
    except Exception as e:
        print(f"Warning: Could not load LoRA adapters: {e}")
        print("Using base model without adapters...")
        model = base
    
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up memory
    del model
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_text.strip()


# ============================================================================
# MAIN BRIDGE FUNCTION
# ============================================================================

def bridge_llm_to_fex(
    model_path: Optional[str] = None,
    prompt: Optional[str] = None,
    postfix_sequence: Optional[str] = None,
    output_path: str = "llm_grammar.json",
    base_model: str = "meta-llama/Llama-3.2-3B",
    verbose: bool = True,
    use_quantization: bool = False,
    device: str = "auto",
) -> Dict:
    """
    Main bridge function: LLM inference → FEX grammar.
    
    Args:
        model_path: Path to fine-tuned model (optional if postfix_sequence provided)
        prompt: PDE description prompt (required if model_path provided)
        postfix_sequence: Direct postfix sequence (bypasses LLM inference)
        output_path: Where to save the grammar JSON
        base_model: Base model name
        verbose: Print detailed output
        use_quantization: Whether to use 4-bit quantization (requires CUDA 13.x)
        device: Device to use ("auto", "cuda", "cpu", "mps")
        
    Returns:
        Generated grammar dictionary
    """
    if verbose:
        print("="*60)
        print("LLM to FEX Grammar Bridge")
        print("="*60)
    
    # Step 1: Get postfix sequence (from LLM or direct input)
    if postfix_sequence:
        if verbose:
            print("\n[1] Using provided postfix sequence:")
            print(f"    {postfix_sequence}")
    elif model_path and prompt:
        if verbose:
            print(f"\n[1] Running LLM inference with model: {model_path}")
            print(f"    Prompt: {prompt[:100]}...")
            print(f"    Quantization: {use_quantization}")
            print(f"    Device: {device}")
        postfix_sequence = run_llm_inference(
            model_path, prompt, base_model,
            use_quantization=use_quantization,
            device=device
        )
        if verbose:
            print(f"    Generated: {postfix_sequence}")
    else:
        raise ValueError("Either model_path+prompt or postfix_sequence must be provided")
    
    # Step 2: Extract operators
    if verbose:
        print("\n[2] Extracting operators from postfix sequence...")
    
    llm_operators = extract_operators_from_postfix(postfix_sequence)
    llm_operators = validate_operators(llm_operators)
    
    if verbose:
        print(f"    Found {len(llm_operators)} unique operators: {sorted(llm_operators)}")
    
    # Step 3: Convert to FEX operators
    if verbose:
        print("\n[3] Mapping to FEX operator names...")
    
    unary_ops, binary_ops = llm_operators_to_fex_grammar(llm_operators)
    
    if verbose:
        print(f"    Unary:  {unary_ops}")
        print(f"    Binary: {binary_ops}")
    
    # Step 4: Generate grammar file
    if verbose:
        print(f"\n[4] Generating FEX grammar file: {output_path}")
    
    metadata = {
        "source": "llm_inference",
        "postfix_sequence": postfix_sequence,
        "llm_operators": sorted(llm_operators),
    }
    
    grammar = generate_fex_grammar_file(unary_ops, binary_ops, output_path, metadata)
    
    if verbose:
        print("\n" + "="*60)
        print("Grammar generated successfully!")
        print(f"Use with FEX: python controller_poisson_grammar.py \\")
        print(f"    --grammar_source llm \\")
        print(f"    --grammar_path {output_path}")
        print("="*60)
    
    return grammar


# ============================================================================
# Convenience Classes for Benchmark Integration
# ============================================================================

class LLMInference:
    """
    Convenience class for LLM inference in benchmark scenarios.
    
    Wraps the model loading and inference logic for easy use in benchmarks.
    """
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-3.2-3B"):
        """
        Initialize LLM inference wrapper.
        
        Args:
            model_path: Path to fine-tuned model (LoRA adapters)
            base_model: Base model name
        """
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}\n"
                "Install with: pip install transformers torch peft"
            )
        
        print(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32
            print("Using MPS (Apple Silicon)")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Using CPU (slow)")
        
        # Load base model
        print(f"Loading base model {self.base_model}...")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        
        # Load LoRA adapters
        print("Loading LoRA adapters...")
        try:
            self.model = PeftModel.from_pretrained(base, self.model_path)
        except Exception as e:
            print(f"Warning: Could not load LoRA adapters: {e}")
            self.model = base
        
        self.model.eval()
        self._loaded = True
        print("Model loaded successfully!")
    
    def predict(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Run inference on a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self._loaded:
            self.load()
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def predict_operators(self, prompt: str, max_new_tokens: int = 64) -> Tuple[Set[str], Set[str], str]:
        """
        Run inference and extract operators from the prediction.
        
        Args:
            prompt: Input prompt (PDE description)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (unary_operators, binary_operators, raw_prediction)
        """
        prediction = self.predict(prompt, max_new_tokens)
        
        # Extract operators
        llm_operators = extract_operators_from_postfix(prediction)
        llm_operators = validate_operators(llm_operators)
        
        # Classify operators
        unary_ops, binary_ops = classify_operators(llm_operators)
        
        return unary_ops, binary_ops, prediction


def classify_operators(operators: Set[str]) -> Tuple[Set[str], Set[str]]:
    """
    Classify operators into unary and binary categories.
    
    Args:
        operators: Set of LLM operator tokens
        
    Returns:
        Tuple of (unary_operators, binary_operators)
    """
    unary = set()
    binary = set()
    
    for op in operators:
        fex_name = LLM_TO_FEX_MAPPING.get(op, op)
        if fex_name in FEX_OPERATOR_TYPES:
            op_type = FEX_OPERATOR_TYPES[fex_name]
            if op_type == "unary":
                unary.add(fex_name)
            else:
                binary.add(fex_name)
    
    return unary, binary


def create_fex_grammar_from_operators(
    unary_ops: Set[str],
    binary_ops: Set[str],
    problem_id: int = 0,
    pde_type: str = "Poisson"
) -> Dict:
    """
    Create a FEX grammar dictionary from operator sets.
    
    Args:
        unary_ops: Set of unary operator names
        binary_ops: Set of binary operator names
        problem_id: Problem identifier
        pde_type: Type of PDE
        
    Returns:
        Grammar dictionary ready for JSON serialization
    """
    grammar = {
        "name": f"LLM_Grammar_Problem_{problem_id}",
        "grammar_type": "llm_generated",
        "version": "1.0",
        "unary_operators": [{"name": op} for op in sorted(unary_ops)],
        "binary_operators": [{"name": op} for op in sorted(binary_ops)],
        "metadata": {
            "source": "llm_inference",
            "problem_id": problem_id,
            "pde_type": pde_type,
        }
    }
    
    return grammar


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bridge LLM inference output to FEX grammar"
    )
    
    # Model inference options
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (LoRA adapters)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model name"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="PDE description prompt (for inference)"
    )
    
    # Direct postfix input (bypasses LLM)
    parser.add_argument(
        "--postfix",
        type=str,
        default=None,
        help="Direct postfix sequence (bypasses LLM inference)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="llm_grammar.json",
        help="Output grammar file path"
    )
    
    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization (requires CUDA 13.x with bitsandbytes)"
    )
    
    # Example mode
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with example postfix sequence"
    )
    
    args = parser.parse_args()
    
    if args.example:
        # Run with example from your dataset
        example_postfix = "const x2 * const const x2 * + *"
        print(f"Running example with postfix: {example_postfix}")
        bridge_llm_to_fex(
            postfix_sequence=example_postfix,
            output_path=args.output,
            verbose=True
        )
    elif args.postfix:
        bridge_llm_to_fex(
            postfix_sequence=args.postfix,
            output_path=args.output,
            verbose=True
        )
    elif args.model_path and args.prompt:
        bridge_llm_to_fex(
            model_path=args.model_path,
            prompt=args.prompt,
            output_path=args.output,
            base_model=args.base_model,
            verbose=True,
            use_quantization=args.quantize,
            device=args.device,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # From direct postfix sequence:")
        print("  python llm_to_fex_bridge.py --postfix 'const x2 sin * exp'")
        print("")
        print("  # From LLM inference (without quantization):")
        print("  python llm_to_fex_bridge.py --model_path ./finetuned_model \\")
        print("      --prompt 'Type: Poisson | RHS: const | Dirichlet: x1=0 const | Solution: '")
        print("")
        print("  # From LLM inference (with 4-bit quantization if CUDA 13.x available):")
        print("  python llm_to_fex_bridge.py --model_path ./finetuned_model \\")
        print("      --prompt 'Type: Poisson | RHS: const | Dirichlet: x1=0 const | Solution: ' \\")
        print("      --quantize")
        print("")
        print("  # Example mode:")
        print("  python llm_to_fex_bridge.py --example")


if __name__ == "__main__":
    main()