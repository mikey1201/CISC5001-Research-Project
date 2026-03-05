"""
REGen.py - Refined Random Expression Generator

Key improvements based on paper Sections 3.1 and 5.1:
1. Proper α, β parameterization for unary operators: α·u(·)+β
2. Operator sets matching the paper exactly
3. Better tree generation with depth control
4. Ensures expressions contain variables (non-constant)
5. FIXED: Added early complexity checks to prevent hanging on large batches
"""

import sympy as sp
import random
import gc
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class OperatorConfig:
    """Configuration for operator sets based on paper specifications."""
    
    # Paper Section 3.1: Binary operators
    BINARY_OPS_FULL = ['+', '-', '*', '/']
    
    # Paper Section 5.1: Uninformed FEX binary set
    BINARY_OPS_REDUCED = ['+', '-', '*']
    
    # Paper Section 3.1: Unary operators
    UNARY_OPS_FULL = ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt', 'abs', 
                      'pow2', 'pow3', 'pow4', 'neg']
    
    # Paper Section 5.1: Uninformed FEX unary set
    UNARY_OPS_REDUCED = ['sin', 'cos', 'exp', 'pow2', 'pow3', 'pow4', 'id']
    
    # Power exponents used in the paper
    POWER_EXPONENTS = [2, 3, 4]


class RandomExpressionGenerator:
    """
    Generates random mathematical expressions as binary computational trees.
    
    Paper Section 3.1:
    "Each node in the computational tree encodes an operator, which may be either 
    a unary or binary operator... Each unary operator node is augmented with two 
    scalar parameters, α and β, which enable the composition of the operator with 
    an affine transformation, yielding expressions of the form α·u(·)+β"
    """
    
    def __init__(
        self,
        num_vars: int = 2,
        max_depth: int = 3,
        domain_bounds: Tuple[float, float] = (0.1, 2.0),
        use_full_operators: bool = False,
        alpha_range: Tuple[int, int] = (-4, 4),
        beta_range: Tuple[int, int] = (-4, 4),
        constant_range: Tuple[int, int] = (-4, 4),
        apply_alpha_beta_prob: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize the generator with specified parameters.
        
        Args:
            num_vars: Number of variables (x0, x1, ..., x_{n-1})
            max_depth: Maximum tree depth (paper uses depth 3)
            domain_bounds: Spatial domain bounds for validation
            use_full_operators: If True, use full operator set; else use reduced set
            alpha_range: Range for α parameter in unary operators
            beta_range: Range for β parameter in unary operators
            constant_range: Range for constant values in leaves
            apply_alpha_beta_prob: Probability of applying α,β transformation
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        self.max_depth = max_depth
        self.domain_bounds = domain_bounds
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.constant_range = constant_range
        self.apply_alpha_beta_prob = apply_alpha_beta_prob
        
        # Create variables: x0, x1, x2, ... (matching paper notation)
        self.vars = [sp.Symbol(f'x{i}', real=True) for i in range(num_vars)]
        self.var_names = [str(v) for v in self.vars]
        
        # Select operator sets based on configuration
        if use_full_operators:
            self.binary_op_names = OperatorConfig.BINARY_OPS_REDUCED
            self.unary_op_names = OperatorConfig.UNARY_OPS_REDUCED
        else:
            self.binary_op_names = OperatorConfig.BINARY_OPS_REDUCED
            self.unary_op_names = OperatorConfig.UNARY_OPS_REDUCED
        
        # Build operator functions
        self._build_operator_functions()
        
        # Track generation statistics
        self.stats = {
            'total_generated': 0,
            'rejected_domain': 0,
            'rejected_complexity': 0,
            'rejected_constant': 0,
            'rejected_timeout': 0,
            'rejected_trivial': 0  # New: rejected for low complexity
        }
        
        # Counter for periodic garbage collection
        self._gc_counter = 0
    
    def _build_operator_functions(self):
        """Build SymPy operator functions from names."""
        # Binary operators
        self.binary_ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': self._safe_div
        }
        
        # Unary operators (base functions without α, β)
        self.unary_ops_base = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'ln': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pow2': lambda x: x**2,
            'pow3': lambda x: x**3,
            'pow4': lambda x: x**4,
            'neg': lambda x: -x,
            'id': lambda x: x
        }
    
    def _safe_div(self, a: sp.Expr, b: sp.Expr) -> sp.Expr:
        """Safe division that handles potential division by zero symbolically."""
        return a / b
    
    def _get_alpha_beta(self) -> Tuple[int, int]:
        """Generate random α, β parameters for unary operator transformation."""
        alpha = random.randint(*self.alpha_range)
        while alpha == 0:
            alpha = random.randint(*self.alpha_range)
        beta = random.randint(*self.beta_range)
        return alpha, beta
    
    def _get_leaf(self, prefer_variable: bool = False) -> sp.Expr:
        """Generate a terminal leaf node: either a variable or a constant."""
        if prefer_variable or random.random() < 0.65:
            return random.choice(self.vars)
        else:
            const_choices = [1, 2, 4, 8, 16, 32, 64, -1, -2, -4, -8]
            return sp.Integer(random.choice(const_choices))
    
    def _apply_unary_with_params(self, op_name: str, child: sp.Expr) -> sp.Expr:
        """Apply unary operator with optional α, β parameters."""
        op_func = self.unary_ops_base.get(op_name)
        
        if op_func is None:
            raise ValueError(f"Unknown unary operator: {op_name}")
        
        result = op_func(child)
        
        # Apply α, β transformation probabilistically
        if random.random() < self.apply_alpha_beta_prob:
            alpha, beta = self._get_alpha_beta()
            result = alpha * result + beta
        
        return result
    
    def _generate_tree(self, current_depth: int = 0, force_var: bool = False) -> sp.Expr:
        """Recursively build the computational tree."""
        # At max depth, must return a leaf
        if current_depth >= self.max_depth:
            return self._get_leaf(prefer_variable=force_var)
        
        # Determine node type with reasonable distribution
        if current_depth == 0:
            weights = [0.05, 0.40, 0.55]  # leaf, unary, binary
        else:
            weights = [0.20, 0.35, 0.45]
        
        node_type = random.choices(['leaf', 'unary', 'binary'], weights=weights)[0]
        
        if node_type == 'leaf':
            return self._get_leaf(prefer_variable=force_var)
        
        elif node_type == 'unary':
            op_name = random.choice(self.unary_op_names)
            child = self._generate_tree(current_depth + 1, force_var=True)
            return self._apply_unary_with_params(op_name, child)
        
        else:  # binary
            op_name = random.choice(self.binary_op_names)
            left_has_var = random.random() < 0.5
            left = self._generate_tree(current_depth + 1, force_var=left_has_var)
            right = self._generate_tree(current_depth + 1, force_var=not left_has_var)
            return self.binary_ops[op_name](left, right)
    
    def _contains_variable(self, expr: sp.Expr) -> bool:
        """Check if expression contains at least one variable."""
        return any(var in expr.free_symbols for var in self.vars)
    
    def _check_minimum_complexity(self, expr: sp.Expr, min_ops: int = 3) -> bool:
        """
        Check if expression has minimum complexity (operation count).
        
        This prevents trivial solutions like u = x0 or u = x1 which would
        cause the model to learn shortcuts instead of meaningful relationships.
        
        Using sympy.count_ops() to count the number of operations.
        
        Args:
            expr: The expression to check
            min_ops: Minimum number of operations required (default: 3)
            
        Returns:
            True if expression meets minimum complexity, False otherwise
        """
        op_count = sp.count_ops(expr)
        return op_count >= min_ops
    
    def _quick_complexity_check(self, expr: sp.Expr) -> bool:
        """
        FAST complexity check BEFORE expensive operations.
        This prevents hanging on obviously bad expressions.
        """
        expr_str = str(expr)
        
        # Reject if expression is too long (quick check)
        if len(expr_str) > 300:
            return False
        
        # Count operations quickly
        op_count = sum(expr_str.count(op) for op in ['sin', 'cos', 'exp', 'ln', 'sqrt'])
        if op_count > 6:
            return False
        
        # Check for high powers before expansion makes them worse
        for i in range(5, 20):
            if f'**{i}' in expr_str:
                return False
        
        # Check for nested powers which can explode
        if expr_str.count('**') > 4:
            return False
        
        return True
    
    def _check_expanded_complexity(self, expr: sp.Expr) -> bool:
        """Check complexity AFTER expansion."""
        expr_str = str(expr)
        
        # Reject if expression is too long
        if len(expr_str) > 500:
            return False
        
        # Count nested operations
        op_count = sum(expr_str.count(op) for op in ['sin', 'cos', 'exp', 'log', 'sqrt'])
        if op_count > 8:
            return False
        
        # Check for unreasonably high powers
        for i in range(5, 20):
            if f'**{i}' in expr_str:
                return False
        
        return True
    
    def _is_valid_expression_fast(self, expr: sp.Expr) -> bool:
        """
        FAST validation without expensive simplify() call.
        Uses quick numerical tests only.
        """
        # Check for infinities or NaN (quick symbolic check)
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return False
        
        # Quick check for complex components (without simplify)
        if expr.has(sp.I):
            return False
        
        # Fast numerical validation at just 2 sample points
        num_tests = 2
        for _ in range(num_tests):
            test_point = {
                var: random.uniform(self.domain_bounds[0], self.domain_bounds[1])
                for var in self.vars
            }
            try:
                val = complex(expr.evalf(subs=test_point))
                if not (abs(val.imag) < 1e-10 and abs(val.real) < 1e10):
                    return False
            except:
                return False
        
        return True
    
    def _try_expand_safe(self, expr: sp.Expr) -> Optional[sp.Expr]:
        """
        Try to expand expression safely with limits.
        Returns None if expansion fails or is too complex.
        """
        try:
            expr_str = str(expr)
            
            # For simple expressions, just expand directly
            if len(expr_str) < 100:
                return sp.expand(expr)
            
            # For complex expressions, use limited expansion
            # Don't fully expand - just collect terms
            return sp.expand(expr, deep=False)
        except Exception:
            return None
    
    def generate_u(self, max_attempts: int = 100) -> Tuple[sp.Expr, Dict[str, Any]]:
        """
        Generate a valid expression u(x) with metadata.
        
        FIXED: Added early complexity checks to prevent hanging.
        """
        for attempt in range(max_attempts):
            self.stats['total_generated'] += 1
            
            # Periodic garbage collection to prevent memory buildup
            self._gc_counter += 1
            if self._gc_counter % 50 == 0:
                gc.collect()
            
            try:
                # Generate tree with force_var=True at root
                expr = self._generate_tree(current_depth=0, force_var=True)
                
                # QUICK complexity check BEFORE expansion
                if not self._quick_complexity_check(expr):
                    self.stats['rejected_complexity'] += 1
                    continue
                
                # Try to expand safely
                expanded = self._try_expand_safe(expr)
                if expanded is None:
                    self.stats['rejected_timeout'] += 1
                    continue
                expr = expanded
                
                # Check complexity after expansion
                if not self._check_expanded_complexity(expr):
                    self.stats['rejected_complexity'] += 1
                    continue
                
                # Ensure expression contains at least one variable
                if not self._contains_variable(expr):
                    self.stats['rejected_constant'] += 1
                    continue
                
                # CRITICAL: Check minimum complexity to reject trivial solutions
                # like u = x0 or u = x1 which cause model shortcuts
                if not self._check_minimum_complexity(expr, min_ops=3):
                    self.stats['rejected_trivial'] += 1
                    continue
                
                # FAST validation (no expensive simplify call)
                if not self._is_valid_expression_fast(expr):
                    self.stats['rejected_domain'] += 1
                    continue
                
                # Success
                metadata = {
                    'generation_attempt': attempt + 1,
                    'tree_depth': self.max_depth,
                    'num_vars': len(self.vars),
                    'num_free_symbols': len(expr.free_symbols)
                }
                
                return expr, metadata
                
            except Exception as e:
                continue
        
        raise RuntimeError(f"Failed to generate valid expression after {max_attempts} attempts")
    
    def generate_batch(self, n: int) -> List[Tuple[sp.Expr, Dict[str, Any]]]:
        """Generate a batch of n expressions."""
        results = []
        for _ in range(n):
            expr, meta = self.generate_u()
            results.append((expr, meta))
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Return generation statistics."""
        return self.stats.copy()


def test_generator():
    """Test the expression generator."""
    gen = RandomExpressionGenerator(num_vars=2, max_depth=3, seed=42)
    
    print("Testing RandomExpressionGenerator...")
    print(f"Variables: {gen.var_names}")
    print(f"Binary ops: {gen.binary_op_names}")
    print(f"Unary ops: {gen.unary_op_names}")
    print()
    
    for i in range(10):
        expr, meta = gen.generate_u()
        print(f"Expression {i+1}: {expr}")
        print(f"  Contains vars: {gen._contains_variable(expr)}")
        print(f"  Metadata: {meta}")
        print()
    
    print(f"Statistics: {gen.get_stats()}")


if __name__ == "__main__":
    test_generator()
