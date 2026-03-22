"""
PDE Operators and Dataset Generation

This module generates training examples for PDE (Partial Differential Equation) 
problems. Each example consists of:
- A symbolic solution expression u(x1, x2, ..., xn)
- The RHS (right-hand side) f derived from the PDE operator applied to u
- A boundary condition g sampled from the solution

Key Features:
- Supports Poisson and Linear Conservation equations
- Each training example has randomized tree depth and variable count
- Expressions are converted to postfix (RPN) notation for ML training
- Maintains full backward compatibility with fixed-parameter mode
- RHS token length is constrained to ensure manageable sequence lengths

Dataset Diversity Improvements:
- Tree depth randomly sampled per example → varying expression complexity
- Variable count randomly sampled per example → varying dimensionality
- This diversity helps ML models generalize better across different PDE structures

RHS Token Constraint:
- RHS expressions exceeding MAX_RHS_TOKENS are rejected and regenerated
- This ensures all training samples have manageable sequence lengths for ML models
"""

import json
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import sympy as sp
import multiprocessing
from functools import partial

# Import from the updated expression generator
from random_expression_generator import (
    generate_expression,
    generate_samples,
    generate_samples_with_metadata,
    get_variables,
    is_constant,
    sample_depth_and_vars,
    # Configuration constants
    MIN_TREE_DEPTH,
    MAX_TREE_DEPTH,
    MIN_NUM_VARIABLES,
    MAX_NUM_VARIABLES,
    # Legacy defaults
    DEFAULT_DEPTH,
    DEFAULT_NUMVARS
)


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Maximum number of tokens allowed in the RHS (right-hand side) expression.
# Samples with RHS exceeding this limit are rejected and regenerated.
# This ensures that all training samples have manageable sequence lengths.
# 
# Rationale:
# - Longer RHS expressions are harder for ML models to process
# - Limits memory usage during training
# - Prevents outlier samples that could destabilize training
# - Value of 120 was chosen to balance expressiveness with trainability
MAX_RHS_TOKENS = 120


# =============================================================================
# MULTIPROCESSING WORKER FUNCTION
# =============================================================================
# This function must be at the top-level (outside the class/function) so it can be
# pickled and sent to worker processes.
def _worker_task(pde_type: str, depth: Optional[int], n: Optional[int], 
                 randomize: bool, max_rhs_tokens: int) -> Tuple[dict, Dict[str, int]]:
    """
    Worker function for multiprocessing. Generates a single valid training example.
    """
    # Each process gets its own random state derived from system entropy
    # or the process ID, ensuring different samples across cores.
    return build_example_with_rhs_limit(
        pde_type, 
        depth=depth, 
        n=n, 
        randomize=randomize, 
        max_rhs_tokens=max_rhs_tokens
    )


# =============================================================================
# PDE OPERATORS
# =============================================================================

def laplacian(u: sp.Expr, n: int) -> sp.Expr:
    """
    Compute the Laplacian of expression u.
    
    The Laplacian is defined as: ∇²u = ∂²u/∂x₁² + ∂²u/∂x₂² + ... + ∂²u/∂xₙ²
    
    Args:
        u: Symbolic expression (the solution)
        n: Number of spatial variables
    
    Returns:
        The Laplacian of u as a symbolic expression
    """
    return sum(sp.diff(u, v, 2) for v in get_variables(n))


def gradient(u: sp.Expr, n: int) -> Tuple[sp.Expr, ...]:
    """
    Compute the gradient of expression u.
    
    The gradient is: ∇u = (∂u/∂x₁, ∂u/∂x₂, ..., ∂u/∂xₙ)
    
    Args:
        u: Symbolic expression
        n: Number of spatial variables
    
    Returns:
        Tuple of partial derivatives
    """
    return tuple(sp.diff(u, v) for v in get_variables(n))


def divergence_term(u: sp.Expr, n: int) -> sp.Expr:
    """
    Compute the divergence of the gradient (for Linear Conservation).
    
    For Linear Conservation: ∇·∇u = ∂u/∂x₁ + ∂u/∂x₂ + ... + ∂u/∂xₙ
    
    This represents a first-order PDE: ∇·∇u = f
    
    Args:
        u: Symbolic expression (the solution)
        n: Number of spatial variables
    
    Returns:
        Sum of first partial derivatives
    """
    return sum(sp.diff(u, v) for v in get_variables(n))


# =============================================================================
# POSTFIX NOTATION CONVERSION
# =============================================================================

def expr_to_postfix(expr: sp.Expr) -> List[str]:
    """
    Convert a SymPy expression to postfix (RPN) token list.
    
    This function standardizes expressions for machine learning training
    by converting them to a consistent token sequence.
    
    Rules:
    1. ALL constants become 'const' (numbers, E, pi, zoo, oo, nan, I, etc.)
    2. Powers with positive integer exponent: x^2 → ['x', '^2']
    3. Division uses '/': a/b → ['a', 'b', '/'], 1/x → ['const', 'x', '/']
    4. sqrt(x) → ['x', 'sqrt']
    5. Functions: sin(x) → ['x', 'sin']
    6. Binary ops: a+b → ['a', 'b', '+'], a*b → ['a', 'b', '*']
    
    Args:
        expr: SymPy symbolic expression
    
    Returns:
        List of tokens in postfix (Reverse Polish Notation) order
    
    Example:
        >>> x = sp.Symbol('x1')
        >>> expr_to_postfix(sp.sin(x) + x**2)
        ['x1', 'sin', 'x1', '^2', '+']
    """
    
    def is_const(e) -> bool:
        """Check if expression is a constant (no free variables)."""
        if e in (sp.E, sp.pi, sp.zoo, sp.oo, sp.nan, sp.I, sp.EulerGamma, sp.GoldenRatio):
            return True
        if e.is_number:
            return True
        return False
    
    def is_negative_power(arg):
        """Check if arg is a negative power (something^(-n) or something^(-p/q))."""
        if not arg.is_Pow:
            return False
        exp = arg.args[1]
        if exp.is_Integer and exp < 0:
            return True
        if exp.is_Rational and exp < 0:
            return True
        return False
    
    def get_negative_power_info(arg):
        """
        For a negative power, return (base, positive_exponent_tokens).
        x^(-1) → (x, [])
        x^(-2) → (x, ['^2'])
        x^(-1/2) → (x, ['sqrt'])
        x^(-3/2) → (x, ['^3', 'sqrt'])
        """
        base, exp = arg.args
        pos_exp = -exp  # Make positive
        
        if pos_exp == 1:
            return (base, [])
        
        if pos_exp.is_Integer:
            return (base, [f'^{int(pos_exp)}'])
        
        if pos_exp.is_Rational:
            num, den = pos_exp.as_numer_denom()
            if den == 2:
                if num == 1:
                    return (base, ['sqrt'])
                else:
                    return (base, [f'^{int(num)}', 'sqrt'])
        
        # Other cases - just return base
        return (base, [])
    
    def to_tokens(e) -> List[str]:
        # 1. Constants → 'const'
        if is_const(e):
            return ['const']
        
        # 2. Symbols (variables x1, x2, x3, etc.)
        if e.is_Symbol:
            return [str(e)]
        
        # 3. Functions: sin, cos, exp, tan, log
        if e.is_Function:
            func_name = e.func.__name__.lower()
            arg_tokens = to_tokens(e.args[0])
            return arg_tokens + [func_name]
        
        # 4. Powers
        if e.is_Pow:
            base, exp = e.args
            
            # If base is const, whole thing is const
            if is_const(base):
                return ['const']
            
            # exp == 0 → const (anything^0 = 1)
            if exp == 0:
                return ['const']
            
            # Positive integer exponent → unary ^N
            if exp.is_Integer and exp > 0:
                return to_tokens(base) + [f'^{int(exp)}']
            
            # Negative exponent → division (handled when this is standalone)
            if exp.is_Integer and exp < 0:
                pos_exp = -int(exp)
                if pos_exp == 1:
                    return ['const'] + to_tokens(base) + ['/']
                else:
                    return ['const'] + to_tokens(base) + [f'^{pos_exp}', '/']
            
            # Rational exponent
            if exp.is_Rational:
                num, den = exp.as_numer_denom()
                
                # Positive rational: x^(1/2) → x sqrt, x^(3/2) → x ^3 sqrt
                if exp > 0:
                    if den == 2:
                        if num == 1:
                            return to_tokens(base) + ['sqrt']
                        else:
                            return to_tokens(base) + [f'^{int(num)}', 'sqrt']
                    # Other denominators - treat as const
                    return ['const']
                
                # Negative rational: x^(-1/2) → const x sqrt /
                if exp < 0:
                    pos_num, pos_den = (-num).as_numer_denom(), den
                    if den == 2:
                        if num == -1:
                            return ['const'] + to_tokens(base) + ['sqrt', '/']
                        else:
                            return ['const'] + to_tokens(base) + [f'^{int(-num)}', 'sqrt', '/']
                    return ['const']
            
            return ['const']
        
        # 5. Multiplication - handle division properly
        if e.is_Mul:
            args = list(e.args)
            numerators = []
            denominators = []
            
            for arg in args:
                if is_const(arg):
                    numerators.append(['const'])
                elif is_negative_power(arg):
                    # This is a denominator term
                    base, exp_tokens = get_negative_power_info(arg)
                    if is_const(base):
                        numerators.append(['const'])  # const^(-n) is still const
                    else:
                        denominators.append(to_tokens(base) + exp_tokens)
                else:
                    numerators.append(to_tokens(arg))
            
            # Build numerator part (multiply all numerators)
            num_result = []
            for num in numerators:
                if num_result:
                    num_result = num_result + num + ['*']
                else:
                    num_result = num
            
            # Build denominator part (multiply all denominators)
            den_result = []
            for den in denominators:
                if den_result:
                    den_result = den_result + den + ['*']
                else:
                    den_result = den
            
            # Combine
            if not denominators:
                return num_result if num_result else ['const']
            if not numerators or not num_result:
                # 1/(denom) → const denom /
                return ['const'] + den_result + ['/']
            
            return num_result + den_result + ['/']
        
        # 6. Addition (n-ary, flattened)
        if e.is_Add:
            args = list(e.args)
            result = to_tokens(args[0])
            for arg in args[1:]:
                result = result + to_tokens(arg) + ['+']
            return result
        
        # 7. Handle any remaining edge cases
        try:
            if e.is_number or not e.free_symbols:
                return ['const']
        except:
            pass
        
        # Last resort
        return [str(e)]
    
    return to_tokens(sp.sympify(expr))


def format_rpn_string(tokens: List[str]) -> str:
    """Join tokens with spaces to create RPN string."""
    return " ".join(tokens)


# =============================================================================
# BOUNDARY CONDITION SAMPLING
# =============================================================================

def sample_boundary(u: sp.Expr, n: int) -> Tuple[str, str, sp.Expr]:
    """
    Sample a random boundary condition on a variable that actually appears in u.
    
    This ensures the boundary condition is meaningful - we only constrain
    variables that participate in the solution expression.
    
    Supported boundary condition types:
    - Dirichlet: u|_Γ = g (prescribes the value of u on boundary)
    - Neumann: ∂u/∂n|_Γ = g (prescribes the normal derivative)
    - Cauchy: Both u and ∂u/∂n prescribed (returns tuple of both)
    
    Args:
        u: Solution expression
        n: Number of spatial variables in the system
    
    Returns:
        Tuple of (bc_type, face_str, g_expr) where:
        - bc_type: 'Dirichlet', 'Neumann', or 'Cauchy'
        - face_str: String describing the boundary location (e.g., "x1=0")
        - g_expr: The boundary value expression (or tuple for Cauchy)
    """
    # Get variables that actually appear in the expression
    active_vars = list(u.free_symbols)
    
    # Fallback: if no variables (constant expression), use first variable from the system
    if not active_vars:
        active_vars = [get_variables(n)[0]]
    
    bc_type = random.choice(['Dirichlet', 'Neumann', 'Cauchy'])
    var = random.choice(active_vars)
    face_value = random.choice([0, 1])
    face_str = f"{str(var)}={face_value}"
    
    if bc_type == 'Dirichlet':
        g = u.subs(var, face_value)
        return bc_type, face_str, g
    if bc_type == 'Neumann':
        g = sp.diff(u, var).subs(var, face_value)
        return bc_type, face_str, g
    
    # Cauchy: return both value and derivative
    g_u = u.subs(var, face_value)
    g_dn = sp.diff(u, var).subs(var, face_value)
    return bc_type, face_str, (g_u, g_dn)


# =============================================================================
# TRAINING EXAMPLE GENERATION
# =============================================================================

def build_example(pde_type: str, 
                  depth: Optional[int] = None, 
                  n: Optional[int] = None,
                  randomize: bool = True,
                  max_rhs_tokens: int = MAX_RHS_TOKENS) -> Tuple[dict, int]:
    """
    Build a single training example for PDE solution prediction.
    
    Each example consists of:
    - A prompt containing the PDE type, RHS (f), and boundary condition
    - A target containing the solution expression in RPN notation
    
    Args:
        pde_type: Either 'Poisson' or 'LinearConservation'
        depth: Tree depth for expression generation (None for randomization)
        n: Number of variables (None for randomization)
        randomize: If True, randomly sample depth and n when not specified
        max_rhs_tokens: Maximum allowed tokens for RHS (samples exceeding this are rejected)
    
    Returns:
        Tuple of (example_dict, rhs_token_count) where:
        - example_dict: Dictionary with keys 'prompt', 'target', 'meta'
        - rhs_token_count: Number of tokens in the RHS expression
    
    Raises:
        ValueError: If pde_type is not 'Poisson' or 'LinearConservation'
    """
    # Determine depth and variable count
    if randomize and (depth is None or n is None):
        sampled_depth, sampled_n = sample_depth_and_vars()
        depth = sampled_depth if depth is None else depth
        n = sampled_n if n is None else n
    else:
        depth = depth if depth is not None else DEFAULT_DEPTH
        n = n if n is not None else DEFAULT_NUMVARS
    
    # Generate a non-constant solution expression
    max_attempts = 100
    for _ in range(max_attempts):
        u = generate_expression(depth, n)
        if not is_constant(u, n):
            break
    else:
        raise RuntimeError(f"Failed to generate non-constant expression after {max_attempts} attempts")
    
    # Apply PDE operator to get RHS
    if pde_type == 'Poisson':
        f_expr = -laplacian(u, n)
    elif pde_type == 'LinearConservation':
        f_expr = divergence_term(u, n)
    else:
        raise ValueError("pde_type must be 'Poisson' or 'LinearConservation'")
    
    # Sample boundary condition
    bc_type, face_str, g_expr = sample_boundary(u, n)
    
    # Convert to postfix notation
    u_tokens = expr_to_postfix(u)
    f_tokens = expr_to_postfix(f_expr)
    
    # Calculate RHS token count
    rhs_token_count = len(f_tokens)
    
    # Build boundary condition RPN string
    if bc_type == 'Cauchy':
        gu, gdn = g_expr
        g_rpn_str = format_rpn_string(expr_to_postfix(gu)) + " | " + format_rpn_string(expr_to_postfix(gdn))
    else:
        g_rpn_str = format_rpn_string(expr_to_postfix(g_expr))
    
    # Build prompt and target
    prompt = (
        f"Type: {('Poisson' if pde_type=='Poisson' else 'LinearConservation')} "
        f"| RHS: {format_rpn_string(f_tokens)} "
        f"| {bc_type}: {face_str} {g_rpn_str} "
        f"| Solution: "
    )
    target = format_rpn_string(u_tokens)
    
    return {
        "prompt": prompt, 
        "target": target, 
        "meta": {
            "u_expr": str(u), 
            "f_expr": str(f_expr), 
            "bc": bc_type, 
            "face": face_str,
            "depth": depth,
            "num_variables": n,
            "variables_used": sorted([str(v) for v in u.free_symbols]),
            "rhs_token_count": rhs_token_count
        }
    }, rhs_token_count, f_expr


def build_example_with_rhs_limit(pde_type: str, 
                                  depth: Optional[int] = None, 
                                  n: Optional[int] = None,
                                  randomize: bool = True,
                                  max_rhs_tokens: int = MAX_RHS_TOKENS,
                                  max_attempts: int = 1000) -> Tuple[dict, Dict[str, int]]:
    """
    Build a training example, regenerating if RHS exceeds token limit.
    
    This function wraps build_example() and ensures that the generated RHS
    expression does not exceed the specified token limit. If it does, the
    sample is discarded and regenerated until a valid sample is found.
    
    Args:
        pde_type: Either 'Poisson' or 'LinearConservation'
        depth: Tree depth for expression generation (None for randomization)
        n: Number of variables (None for randomization)
        randomize: If True, randomly sample depth and n when not specified
        max_rhs_tokens: Maximum allowed tokens for RHS
        max_attempts: Maximum number of regeneration attempts
    
    Returns:
        Tuple of (example_dict, stats) where:
        - example_dict: The accepted example with keys 'prompt', 'target', 'meta'
        - stats: Dictionary with 'attempts', 'rhs_tokens', 'rejected_count'
    
    Raises:
        RuntimeError: If max_attempts exceeded without finding valid sample
    """
    attempts = 0
    rejected_count = 0
    
    while attempts < max_attempts:
        attempts += 1
        ex, rhs_token_count, f_expr = build_example(pde_type, depth, n, randomize, max_rhs_tokens)
        

def build_example_with_rhs_limit(pde_type: str, 
                                  depth: Optional[int] = None, 
                                  n: Optional[int] = None,
                                  randomize: bool = True,
                                  max_rhs_tokens: int = MAX_RHS_TOKENS,
                                  max_attempts: int = 1000) -> Tuple[dict, Dict[str, int]]:
    attempts = 0
    rejected_count = 0
    
    while attempts < max_attempts:
        attempts += 1
        # Unpack the new return signature
        ex, rhs_token_count, f_expr = build_example(pde_type, depth, n, randomize, max_rhs_tokens)
        
        # Reject 60% of depth-1 cases where RHS is zero (trivial PDE)
        if ex["meta"]["depth"] == 1 and f_expr.is_zero:
            if random.random() < 0.6:
                rejected_count += 1
                continue

        if rhs_token_count <= max_rhs_tokens:
            stats = {
                'attempts': attempts,
                'rhs_tokens': rhs_token_count,
                'rejected_count': rejected_count
            }
            return ex, stats
        
        rejected_count += 1
    
    raise RuntimeError(
        f"Failed to generate sample with RHS <= {max_rhs_tokens} tokens after {max_attempts} attempts. "
        f"Consider increasing max_rhs_tokens or reducing expression complexity."
    )


def generate_dataset(num_per_pde: int, 
                     depth: Optional[int] = None, 
                     n: Optional[int] = None,
                     out_jsonl: str = "pde_dataset.jsonl",
                     randomize: bool = True,
                     show_stats: bool = True,
                     max_rhs_tokens: int = MAX_RHS_TOKENS) -> Dict[str, Any]:
    """
    Generate the full PDE dataset with RHS token length constraint.
    Uses multiprocessing to parallelize generation across CPU cores.
    """
    pde_types = ['Poisson', 'LinearConservation']
    total = num_per_pde * len(pde_types)
    
    # Track statistics
    depth_counts = Counter()
    var_counts = Counter()
    bc_counts = Counter()
    rhs_token_counts: List[int] = []
    examples_metadata = []
    
    # Track rejection statistics
    total_attempts = 0
    total_rejected = 0
    
    # Create a partial function with fixed arguments for the worker
    worker_func = partial(_worker_task, 
                          depth=depth, 
                          n=n, 
                          randomize=randomize, 
                          max_rhs_tokens=max_rhs_tokens)
    
    # Prepare tasks: repeat the pde type for the number of samples needed
    tasks = []
    for pde in pde_types:
        for _ in range(num_per_pde):
            tasks.append(pde)
    
    print(f"Starting generation of {total} examples using {multiprocessing.cpu_count()} CPU cores...")
    
    with open(out_jsonl, 'w', encoding='utf-8') as fh:
        # Create a pool of worker processes
        # processes=None means it uses os.cpu_count() (all your logical cores)
        with multiprocessing.Pool(processes=None) as pool:
            # imap_unordered allows processing results as soon as they are ready
            # This is faster than waiting for the whole batch
            for result in pool.imap_unordered(worker_func, tasks):
                ex, gen_stats = result
                
                # Write to file immediately to save memory
                fh.write(json.dumps({"prompt": ex["prompt"], "target": ex["target"]}) + "\n")
                
                # Track statistics
                depth_counts[ex["meta"]["depth"]] += 1
                var_counts[ex["meta"]["num_variables"]] += 1
                bc_counts[ex["meta"]["bc"]] += 1
                rhs_token_counts.append(ex["meta"]["rhs_token_count"])
                examples_metadata.append(ex["meta"])
                
                total_attempts += gen_stats['attempts']
                total_rejected += gen_stats['rejected_count']
    
    print(f"\rWrote {total} examples to {out_jsonl}")
    
    if show_stats:
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        # RHS Token Statistics
        print(f"\nRHS Token Length Statistics:")
        print(f"  Limit: {max_rhs_tokens} tokens")
        print(f"  Mean: {sum(rhs_token_counts)/len(rhs_token_counts):.1f} tokens")
        print(f"  Max: {max(rhs_token_counts)} tokens")
        print(f"  Min: {min(rhs_token_counts)} tokens")
        print(f"  Rejected: {total_rejected}/{total_attempts} ({total_rejected/total_attempts*100:.1f}%)")
        
        # RHS token distribution histogram
        print(f"\n  Token Distribution:")
        bins = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 120)]
        for low, high in bins:
            count = sum(1 for t in rhs_token_counts if low <= t <= high)
            pct = count / total * 100
            bar = "█" * int(pct / 100 * 40)
            print(f"    {low:3d}-{high:3d}: {count:4d} ({pct:5.1f}%) {bar}")
        
        print(f"\nTree Depth Distribution:")
        for d in sorted(depth_counts.keys()):
            count = depth_counts[d]
            pct = count / total * 100
            bar = "█" * int(pct / 100 * 40)
            print(f"  Depth {d}: {count:4d} ({pct:5.1f}%) {bar}")
        
        print(f"\nVariable Count Distribution:")
        for v in sorted(var_counts.keys()):
            count = var_counts[v]
            pct = count / total * 100
            bar = "█" * int(pct / 100 * 40)
            print(f"  {v} vars: {count:4d} ({pct:5.1f}%) {bar}")
        
        print(f"\nBoundary Condition Types:")
        for bc in sorted(bc_counts.keys()):
            count = bc_counts[bc]
            pct = count / total * 100
            print(f"  {bc}: {count:4d} ({pct:5.1f}%)")
        
        print("\n" + "-" * 60)
        print("SAMPLE EXPRESSIONS:")
        print("-" * 60)
        for i, meta in enumerate(examples_metadata[:5]):
            print(f"\n{i+1}. Depth={meta['depth']}, Vars={meta['num_variables']}, RHS tokens={meta['rhs_token_count']}")
            print(f"   Variables used: {meta['variables_used']}")
            print(f"   u = {meta['u_expr']}")
            print(f"   f = {meta['f_expr']}")
            print(f"   BC: {meta['bc']} at {meta['face']}")
    
        
    return {
        "total_examples": total,
        "depth_distribution": dict(depth_counts),
        "var_distribution": dict(var_counts),
        "bc_distribution": dict(bc_counts),
        "rhs_token_distribution": rhs_token_counts,
        "rejection_stats": {
            "total_attempts": total_attempts,
            "total_rejected": total_rejected,
            "rejection_rate": total_rejected / total_attempts if total_attempts > 0 else 0
        },
        "examples_metadata": examples_metadata
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PDE dataset for LLM training')
    parser.add_argument('--samples', type=int, default=100, 
                        help='Number of samples per PDE type (default: 100)')
    parser.add_argument('--depth', type=int, default=None, 
                        help='Fixed tree depth (omit for randomization)')
    parser.add_argument('--vars', type=int, default=None, 
                        help='Fixed number of variables (omit for randomization)')
    parser.add_argument('--output', type=str, default='pde_dataset.jsonl', 
                        help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-rhs-tokens', type=int, default=MAX_RHS_TOKENS,
                        help=f'Maximum RHS token count (default: {MAX_RHS_TOKENS})')
    parser.add_argument('--validate', action='store_true', 
                        help='Run validation with sample verification')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.validate:
        # Run validation mode
        print("=" * 60)
        print("VALIDATION MODE - Testing RHS Token Limit")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Depth range: [{MIN_TREE_DEPTH}, {MAX_TREE_DEPTH}]")
        print(f"  Variable range: [{MIN_NUM_VARIABLES}, {MAX_NUM_VARIABLES}]")
        print(f"  Max RHS tokens: {args.max_rhs_tokens}")
        print()
        
        stats = generate_dataset(
            args.samples, 
            depth=args.depth, 
            n=args.vars,
            out_jsonl=args.output,
            randomize=(args.depth is None and args.vars is None),
            show_stats=True,
            max_rhs_tokens=args.max_rhs_tokens
        )
        
        # Verify no RHS exceeds limit
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        max_observed = max(stats['rhs_token_distribution'])
        all_valid = all(t <= args.max_rhs_tokens for t in stats['rhs_token_distribution'])
        
        print(f"\n✓ All RHS token counts <= {args.max_rhs_tokens}: {all_valid}")
        print(f"✓ Maximum observed RHS tokens: {max_observed}")
        
        if all_valid:
            print(f"\n✓ VALIDATION PASSED: All {stats['total_examples']} samples respect the {args.max_rhs_tokens}-token limit")
        else:
            print(f"\n✗ VALIDATION FAILED: Some samples exceed the limit!")
            violations = [t for t in stats['rhs_token_distribution'] if t > args.max_rhs_tokens]
            print(f"  Violations: {len(violations)} samples with tokens {violations}")
    else:
        # Normal generation mode
        randomize = (args.depth is None and args.vars is None)
        if randomize:
            print(f"Generating {args.samples} samples per PDE type with RANDOMIZED depth and variables")
        else:
            print(f"Generating {args.samples} samples per PDE type with FIXED depth={args.depth}, vars={args.vars}")
        print(f"Max RHS tokens: {args.max_rhs_tokens}")
        
        generate_dataset(
            args.samples, 
            depth=args.depth, 
            n=args.vars,
            out_jsonl=args.output,
            randomize=randomize,
            show_stats=True,
            max_rhs_tokens=args.max_rhs_tokens
        )
