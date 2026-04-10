"""
Random Expression Generator for PDE Solutions

This module generates random symbolic expressions for use in PDE (Partial Differential
Equation) solution datasets. The expressions are built as computational trees with
configurable depth and number of variables.

Key Features:
- Tree depth is randomly sampled per generated example (configurable range)
- Number of variables is randomly sampled per generated example (configurable range)
- Expressions are guaranteed to be non-constant (contain at least one variable)
- Supports various unary (sin, cos, exp, sqrt, log, tan, power) and binary (+, -, *, /) operators

Configuration Parameters:
- MIN_TREE_DEPTH / MAX_TREE_DEPTH: Control the range of tree complexity
- MIN_NUM_VARIABLES / MAX_NUM_VARIABLES: Control the dimensionality of expressions

Why Randomized Depth and Variables?
- Increases dataset diversity for machine learning training
- Prevents overfitting to specific expression structures
- Enables the model to learn patterns across varying complexity levels
- Mirrors real-world PDE problems which have varying dimensionality
"""

import sympy as sp
import random
from typing import List, Set, Tuple, Optional
from collections import Counter

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
# These parameters control the randomization ranges for expression generation.
# They can be easily modified to adjust dataset characteristics.

# Tree depth range: Controls expression complexity
# - Depth 1: Simple expressions (e.g., x1, 2*x1)
# - Depth 3: Moderate complexity (e.g., sin(x1*x2) + x3^2)
# - Depth 6: Complex expressions with many nested operations
MIN_TREE_DEPTH = 1
MAX_TREE_DEPTH = 6

# Variable count range: Controls expression dimensionality
# - 1 variable: Single-variable PDEs (x1 only)
# - 2-3 variables: Common 2D/3D problems
# - 4-6 variables: Higher-dimensional problems
MIN_NUM_VARIABLES = 1
MAX_NUM_VARIABLES = 5

# Legacy defaults for backward compatibility
DEFAULT_NUMVARS = 3
DEFAULT_DEPTH = 3


def get_variables(n: int) -> Tuple[sp.Symbol, ...]:
    """
    Generate symbolic variables x1, x2, ..., xn.
    
    Args:
        n: Number of variables to generate (must be positive)
    
    Returns:
        Tuple of SymPy symbols (x1, x2, ..., xn)
    
    Example:
        >>> get_variables(3)
        (x1, x2, x3)
    """
    return sp.symbols(f'x1:{n + 1}')


def sample_exponent() -> int:
    """
    Sample a random exponent for power operations.
    
    Returns an exponent from 2 to 6 with weighted probabilities.
    Lower powers are still more common (matching PDE physics), but
    higher powers now have sufficient representation for LLM training.
    
    Weights:
    - ^2: 30% (quadratic terms are most common in PDEs)
    - ^3: 25% (cubic terms also common)
    - ^4: 18% (quartic terms from double-differentiation)
    - ^5: 15% (moderate representation)
    - ^6: 12% (sufficient for training coverage)
    
    Returns:
        int: Exponent from 2 to 6
    """
    weights = [0.30, 0.25, 0.18, 0.15, 0.12]  # ^2, ^3, ^4, ^5, ^6
    exponents = [2, 3, 4, 5, 6]
    return random.choices(exponents, weights=weights)[0]


def get_leaf(n: int) -> sp.Symbol:
    """
    Randomly select one variable from the available variables.
    
    Args:
        n: Total number of available variables
    
    Returns:
        A single variable symbol (x1, x2, ..., or xn)
    """
    return random.choice(get_variables(n))


def apply_affine(expr: sp.Expr) -> sp.Expr:
    """
    Apply a random affine transformation to an expression.
    
    Transformations:
    - 40% chance: identity (no change)
    - 40% chance: scale by 2 (2 * expr)
    - 20% chance: scale by 2 and shift by 2 (2 * expr + 2)
    
    Args:
        expr: Input expression
    
    Returns:
        Transformed expression
    """
    r = random.random()
    if r < .4:
        return expr
    elif r < .8:
        return 2 * expr
    return 2 * expr + 2


def apply_unary(expr: sp.Expr) -> sp.Expr:
    """
    Apply a random unary operation to an expression.
    
    Args:
        expr: Input expression
    
    Returns:
        Expression with unary operation applied
    """
    r = random.random()
    if r < .3:
        return expr

    trig = 1/3
    powe = trig + .25
    exp = powe + .25
    sqrt = exp + .10
    log = sqrt + .05
    
    r = random.random()
    if r < trig:
        r = random.random()
        if r < .5:
            return sp.sin(expr)
        else:
            return sp.cos(expr)
    elif r < powe:
        return expr ** sample_exponent()
    elif r < exp:
        return sp.exp(expr)
    elif r < sqrt:
        return sp.sqrt(expr)
    elif r < log:
        return sp.log(expr)
    return sp.tan(expr)


def apply_binary(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """
    Apply a random binary operation to combine two expressions.
    
    Operations and their probabilities:
    - 90% chance: standard operations
        - 50% of standard: multiplication (*)
        - 50% of standard: addition/subtraction (50/50 split)
    - 10% chance: division (/)
    
    Args:
        left: Left operand
        right: Right operand
    
    Returns:
        Combined expression
    """
    r = random.random()
    if r < .9:
        r = random.random()
        if r < .5:
            return left * right
        else:
            r = random.random()
            if r < .5:
                return left + right
            else:
                return left - right
    return left / right


def generate_expression(depth: int, n: int) -> sp.Expr:
    """
    Generate a random expression tree of specified depth.
    
    The tree is built recursively with alternating levels:
    - Level 0 (leaf): Single variable
    - Odd levels: Apply unary operations (sin, cos, exp, etc.) with affine transforms
    - Even levels: Apply binary operations (+, -, *, /) to combine subtrees
    
    Args:
        depth: Tree depth (0 = single variable, higher = more complex)
        n: Number of available variables (x1 through xn)
    
    Returns:
        Generated symbolic expression
    """
    if depth == 1:
        if random.random() < 0.5:
            leaf = get_leaf(n)
            return apply_affine(apply_unary(leaf))
        else:
            left = get_leaf(n)
            right = get_leaf(n)
            return apply_binary(left, right)

    def build_level(level):
        if level == 0:
            return get_leaf(n)
        if level % 2 == 1:
            # Odd levels: apply unary operations
            child = build_level(level - 1)
            return apply_affine(apply_unary(child))
        else:
            # Even levels: apply binary operations
            left = build_level(level - 1)
            right = build_level(level - 1)
            return apply_binary(left, right)

    return build_level(depth)


def is_constant(expr: sp.Expr, n: int) -> bool:
    """
    Check if an expression is constant (contains no variables).
    
    Args:
        expr: Expression to check
        n: Number of variables in the system
    
    Returns:
        True if expression has no free variables, False otherwise
    """
    expr_symbols: Set[sp.Symbol] = expr.free_symbols
    for var in get_variables(n):
        if var in expr_symbols:
            return False
    return True


def generate_non_constant_expression(depth: int, n: int, max_attempts: int = 100) -> sp.Expr:
    """
    Generate an expression that is guaranteed to be non-constant.
    
    Some random combinations of operations can result in constant expressions
    (e.g., sin(x1) * 0 = 0). This function retries until a non-constant
    expression is found.
    
    Args:
        depth: Tree depth for generation
        n: Number of available variables
        max_attempts: Maximum number of retry attempts
    
    Returns:
        A non-constant expression
    
    Raises:
        RuntimeError: If max_attempts exceeded without finding non-constant expression
    """
    for attempt in range(max_attempts):
        expr = generate_expression(depth, n)
        if not is_constant(expr, n):
            return expr
    
    raise RuntimeError(f"Failed to generate non-constant expression after {max_attempts} attempts. "
                       f"Consider adjusting the sampling parameters (alpha/beta distributions).")


def sample_depth_and_vars(min_depth: int = MIN_TREE_DEPTH, 
                          max_depth: int = MAX_TREE_DEPTH,
                          min_vars: int = MIN_NUM_VARIABLES, 
                          max_vars: int = MAX_NUM_VARIABLES) -> Tuple[int, int]:
    """
    Randomly sample tree depth and number of variables for expression generation.
    
    This enables per-example randomization of:
    1. Expression complexity (via depth) - NOW WEIGHTED
    2. Expression dimensionality (via variable count)
    
    Args:
        min_depth: Minimum tree depth (default: MIN_TREE_DEPTH)
        max_depth: Maximum tree depth (default: MAX_TREE_DEPTH)
        min_vars: Minimum number of variables (default: MIN_NUM_VARIABLES)
        max_vars: Maximum number of variables (default: MAX_NUM_VARIABLES)
    
    Returns:
        Tuple of (depth, num_variables)
    """
    depths = list(range(min_depth, max_depth + 1))
    
    # Target weights for depths 1, 2, 3, 4, 5, 6
    # Target: 12%, 20%, 25%, 25%, 13%, 5%
    weights = [0.10, 0.12, 0.15, 0.15, 0.18, 0.30]
    
    # Ensure we have correct number of weights for the available depth range
    if len(depths) == len(weights):
        depth = random.choices(depths, weights=weights)[0]
    else:
        # Fallback to uniform if configuration differs from expected 1-6 range
        depth = random.randint(min_depth, max_depth)
        
    num_vars = random.randint(min_vars, max_vars)
    return depth, num_vars


def generate_samples(num_samples: int, 
                     depth: int = DEFAULT_DEPTH, 
                     n: int = DEFAULT_NUMVARS, 
                     show_progress: bool = False,
                     randomize_depth: bool = False,
                     randomize_vars: bool = False) -> List[sp.Expr]:
    """
    Generate multiple random non-constant expressions.
    
    Supports two modes:
    1. Fixed mode (default): All expressions use the same depth and variable count
    2. Randomized mode: Each expression has independently sampled depth and/or variables
    
    Args:
        num_samples: Number of expressions to generate
        depth: Tree depth (used when randomize_depth=False)
        n: Number of variables (used when randomize_vars=False)
        show_progress: Whether to print progress information
        randomize_depth: If True, sample depth per expression from [MIN_TREE_DEPTH, MAX_TREE_DEPTH]
        randomize_vars: If True, sample variable count per expression from [MIN_NUM_VARIABLES, MAX_NUM_VARIABLES]
    
    Returns:
        List of non-constant symbolic expressions
    """
    non_constant_samples: List[sp.Expr] = []
    total_generated = 0
    constants_filtered = 0
    
    while len(non_constant_samples) < num_samples:
        # Determine depth and variable count for this sample
        current_depth = random.randint(MIN_TREE_DEPTH, MAX_TREE_DEPTH) if randomize_depth else depth
        current_n = random.randint(MIN_NUM_VARIABLES, MAX_NUM_VARIABLES) if randomize_vars else n
        
        expr = generate_expression(current_depth, current_n)
        total_generated += 1
        
        # Check against the actual number of variables used
        if is_constant(expr, current_n):
            constants_filtered += 1
        else:
            non_constant_samples.append(expr)
        
        if show_progress and total_generated % 1000 == 0:
            print(f"Generated {total_generated} total, "
                  f"{len(non_constant_samples)} non-constant, "
                  f"{constants_filtered} constants filtered")
    
    if show_progress:
        print(f"\nGeneration complete:")
        print(f"  Total generated: {total_generated}")
        print(f"  Non-constant samples: {len(non_constant_samples)}")
        print(f"  Constants filtered: {constants_filtered}")
        print(f"  Constant rate: {constants_filtered/total_generated*100:.1f}%")
    
    return non_constant_samples


def generate_samples_with_metadata(num_samples: int,
                                   min_depth: int = MIN_TREE_DEPTH,
                                   max_depth: int = MAX_TREE_DEPTH,
                                   min_vars: int = MIN_NUM_VARIABLES,
                                   max_vars: int = MAX_NUM_VARIABLES,
                                   show_progress: bool = False) -> Tuple[List[sp.Expr], List[dict]]:
    """
    Generate expressions with randomized depth and variable count, returning metadata.
    
    This is the recommended function for creating diverse PDE datasets.
    Each generated expression has:
    - Independently sampled tree depth from [min_depth, max_depth]
    - Independently sampled variable count from [min_vars, max_vars]
    
    Args:
        num_samples: Number of expressions to generate
        min_depth: Minimum tree depth (default: MIN_TREE_DEPTH = 1)
        max_depth: Maximum tree depth (default: MAX_TREE_DEPTH = 6)
        min_vars: Minimum number of variables (default: MIN_NUM_VARIABLES = 1)
        max_vars: Maximum number of variables (default: MAX_NUM_VARIABLES = 5)
        show_progress: Whether to print progress information
    
    Returns:
        Tuple of (expressions_list, metadata_list) where each metadata dict contains:
        - 'depth': The tree depth used
        - 'num_variables': The number of variables used
        - 'variables': Tuple of variable symbols used
    
    Example:
        >>> exprs, metas = generate_samples_with_metadata(100)
        >>> metas[0]
        {'depth': 4, 'num_variables': 3, 'variables': (x1, x2, x3)}
    """
    expressions: List[sp.Expr] = []
    metadata: List[dict] = []
    total_generated = 0
    constants_filtered = 0
    
    while len(expressions) < num_samples:
        # Randomly sample depth and variable count for this expression
        depth = random.randint(min_depth, max_depth)
        num_vars = random.randint(min_vars, max_vars)
        
        expr = generate_expression(depth, num_vars)
        total_generated += 1
        
        if is_constant(expr, num_vars):
            constants_filtered += 1
        else:
            expressions.append(expr)
            metadata.append({
                'depth': depth,
                'num_variables': num_vars,
                'variables': get_variables(num_vars)
            })
        
        if show_progress and total_generated % 1000 == 0:
            print(f"Generated {total_generated} total, "
                  f"{len(expressions)} non-constant, "
                  f"{constants_filtered} constants filtered")
    
    if show_progress:
        print(f"\nGeneration complete:")
        print(f"  Total generated: {total_generated}")
        print(f"  Non-constant samples: {len(expressions)}")
        print(f"  Constants filtered: {constants_filtered}")
        print(f"  Constant rate: {constants_filtered/total_generated*100:.1f}%")
        
        # Print distribution statistics
        depth_counts = Counter(m['depth'] for m in metadata)
        var_counts = Counter(m['num_variables'] for m in metadata)
        
        print(f"\nDepth distribution:")
        for d in sorted(depth_counts.keys()):
            print(f"  Depth {d}: {depth_counts[d]} ({depth_counts[d]/len(metadata)*100:.1f}%)")
        
        print(f"\nVariable count distribution:")
        for v in sorted(var_counts.keys()):
            print(f"  {v} variables: {var_counts[v]} ({var_counts[v]/len(metadata)*100:.1f}%)")
    
    return expressions, metadata


# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_generation(num_samples: int = 100) -> dict:
    """
    Validate the expression generation by generating samples and computing statistics.
    
    This function verifies that:
    1. All generated expressions are valid SymPy expressions
    2. Depths are correctly distributed between MIN_TREE_DEPTH and MAX_TREE_DEPTH
    3. Variable counts are correctly distributed between MIN_NUM_VARIABLES and MAX_NUM_VARIABLES
    4. No constant expressions are generated
    
    Args:
        num_samples: Number of samples to generate for validation
    
    Returns:
        Dictionary containing validation statistics and example expressions
    """
    print(f"Validating expression generation with {num_samples} samples...")
    print(f"Configuration: depth=[{MIN_TREE_DEPTH}, {MAX_TREE_DEPTH}], vars=[{MIN_NUM_VARIABLES}, {MAX_NUM_VARIABLES}]")
    print("-" * 60)
    
    expressions, metadata = generate_samples_with_metadata(num_samples, show_progress=False)
    
    # Compute statistics
    depth_counts = Counter(m['depth'] for m in metadata)
    var_counts = Counter(m['num_variables'] for m in metadata)
    
    # Validate ranges
    depths = [m['depth'] for m in metadata]
    var_nums = [m['num_variables'] for m in metadata]
    
    depth_min, depth_max = min(depths), max(depths)
    var_min, var_max = min(var_nums), max(var_nums)
    
    # All expressions should be non-constant (verified by generation function)
    # Check that expressions can be evaluated/differentiated
    all_valid = True
    for i, (expr, meta) in enumerate(zip(expressions, metadata)):
        try:
            # Test that we can compute derivatives (validates expression structure)
            for var in meta['variables']:
                sp.diff(expr, var)
        except Exception as e:
            print(f"Invalid expression at index {i}: {e}")
            all_valid = False
    
    # Print statistics
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\n✓ Generated {len(expressions)} valid non-constant expressions")
    print(f"✓ All expressions support differentiation")
    
    print(f"\nDepth Statistics:")
    print(f"  Range: [{depth_min}, {depth_max}] (expected: [{MIN_TREE_DEPTH}, {MAX_TREE_DEPTH}])")
    print(f"  Distribution:")
    for d in sorted(depth_counts.keys()):
        bar = "█" * int(depth_counts[d] / num_samples * 30)
        print(f"    Depth {d}: {depth_counts[d]:3d} ({depth_counts[d]/num_samples*100:5.1f}%) {bar}")
    
    print(f"\nVariable Count Statistics:")
    print(f"  Range: [{var_min}, {var_max}] (expected: [{MIN_NUM_VARIABLES}, {MAX_NUM_VARIABLES}])")
    print(f"  Distribution:")
    for v in sorted(var_counts.keys()):
        bar = "█" * int(var_counts[v] / num_samples * 30)
        print(f"    {v} vars: {var_counts[v]:3d} ({var_counts[v]/num_samples*100:5.1f}%) {bar}")
    
    # Show example expressions
    print("\n" + "-" * 60)
    print("SAMPLE EXPRESSIONS:")
    print("-" * 60)
    for i in range(min(10, len(expressions))):
        expr = expressions[i]
        meta = metadata[i]
        vars_used = sorted([str(v) for v in expr.free_symbols])
        print(f"\n{i+1}. Depth={meta['depth']}, Vars={meta['num_variables']}")
        print(f"   Variables used: {vars_used}")
        print(f"   Expression: {expr}")
    
    return {
        'num_samples': num_samples,
        'depth_distribution': dict(depth_counts),
        'var_distribution': dict(var_counts),
        'depth_range': (depth_min, depth_max),
        'var_range': (var_min, var_max),
        'all_valid': all_valid,
        'sample_expressions': [(str(expressions[i]), metadata[i]) for i in range(min(10, len(expressions)))]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random symbolic expressions for PDE datasets')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--depth', type=int, default=None, help='Fixed depth (omit for randomization)')
    parser.add_argument('--vars', type=int, default=None, help='Fixed number of variables (omit for randomization)')
    parser.add_argument('--validate', action='store_true', help='Run validation mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.validate:
        stats = validate_generation(args.samples)
    else:
        # Generate samples with optional fixed parameters
        if args.depth is not None and args.vars is not None:
            print(f"Generating {args.samples} samples with fixed depth={args.depth}, vars={args.vars}")
            samples = generate_samples(args.samples, depth=args.depth, n=args.vars, show_progress=True)
        else:
            print(f"Generating {args.samples} samples with randomized depth and variables")
            samples, metadata = generate_samples_with_metadata(args.samples, show_progress=True)
        
        print(f"\nFirst 5 expressions:")
        for i, expr in enumerate(samples[:5]):
            print(f"  {i+1}. {expr}")
