"""
RPNC.py - RPN (Reverse Polish Notation) Converter

Converts SymPy expressions to postfix notation matching the paper's format.

Paper Figure 3 shows the expected format:
- Input: "Type: Poisson | RHS: const x0 * | Dirichlet: x0=0 const | Solution: || x0 ^2 const *"
- Format: [PDE_Type] | RHS: [f_ops] | [BC_Type]: [boundary] [g_ops] | Solution: || [u_ops]
- Constants are represented as 'const'
- Power notation: x0^2 becomes 'x0' '^2'
- The '||' appears to be a separator before the solution

Key observations from paper:
1. Variables: x0, x1, x2, x3, ... (with numeric indices)
2. Constants: represented as 'const' (not their numeric values)
3. Operators: +, -, *, /, sin, cos, exp, ln, sqrt, abs, ^2, ^3, ^4
4. Format preserves operator sequence for LLM training
"""

import sympy as sp
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class RPNConfig:
    """Configuration for RPN conversion matching paper format."""
    
    # Token representations
    CONST_TOKEN = 'const'
    SEPARATOR_TOKEN = '||'
    
    # Power tokens (paper uses ^2, ^3, ^4)
    # Extended to handle common powers that may appear
    POWER_TOKENS = {
        2: '^2',
        3: '^3', 
        4: '^4',
        5: '^5',
        6: '^6',
        7: '^7',
        8: '^8',
    }
    
    # Maximum power to convert to token (larger powers use general ^ notation)
    MAX_POWER_TOKEN = 8
    
    # Function name mapping (SymPy -> paper format)
    FUNCTION_MAP = {
        'sin': 'sin',
        'cos': 'cos',
        'tan': 'tan',
        'exp': 'exp',
        'log': 'ln',  # SymPy uses 'log' for natural log
        'sqrt': 'sqrt',
        'Abs': 'abs',
        'abs': 'abs'
    }


class RPNConverter:
    """
    Converts SymPy expressions to RPN token lists.
    
    Paper Section 3.2:
    "The symbolic expressions are tokenized and converted into postfix notation 
    (Reverse Polish Notation). This format is chosen because it aligns well with 
    the sequential input requirements of LLMs and eliminates the need for parentheses, 
    simplifying the parsing process."
    """
    
    def __init__(self):
        self.config = RPNConfig()
    
    def to_rpn(self, expr: sp.Expr) -> List[str]:
        """
        Convert a SymPy expression to RPN token list.
        
        Args:
            expr: SymPy expression
            
        Returns:
            List of RPN tokens
        """
        # Simplify first
        expr = sp.expand(expr)
        
        # Check for invalid expressions
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I):
            raise ValueError("Expression contains infinity or complex numbers")
        
        return self._convert(expr)
    
    def _convert(self, expr: sp.Expr) -> List[str]:
        """Recursive conversion to RPN."""
        
        # 1. Symbol (variable)
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        
        # 2. Numeric constant (including integers, floats, rationals)
        if expr.is_Number:
            return [self.config.CONST_TOKEN]
        
        # 3. Euler's number e
        if expr == sp.E:
            return [self.config.CONST_TOKEN]
        
        # 4. Pi (treat as constant)
        if expr == sp.pi:
            return [self.config.CONST_TOKEN]
        
        # 5. Addition
        if isinstance(expr, sp.Add):
            args = list(expr.args)
            result = self._convert(args[0])
            for arg in args[1:]:
                result.extend(self._convert(arg))
                result.append('+')
            return result
        
        # 6. Multiplication
        if isinstance(expr, sp.Mul):
            args = list(expr.args)
            result = self._convert(args[0])
            for arg in args[1:]:
                result.extend(self._convert(arg))
                result.append('*')
            return result
        
        # 7. Power / Exponentiation
        if isinstance(expr, sp.Pow):
            base, exp = expr.args
            
            # Handle integer exponents (power tokens)
            if exp.is_Integer and int(exp) in self.config.POWER_TOKENS:
                base_tokens = self._convert(base)
                return base_tokens + [self.config.POWER_TOKENS[int(exp)]]
            
            # Handle negative integer exponents
            if exp.is_Integer and int(exp) < 0:
                # x^(-n) = 1/x^n
                pos_exp = -int(exp)
                if pos_exp in self.config.POWER_TOKENS:
                    base_tokens = self._convert(base)
                    return [self.config.CONST_TOKEN] + base_tokens + \
                           [self.config.POWER_TOKENS[pos_exp]] + ['/']
            
            # General exponent: base ^ exp
            base_tokens = self._convert(base)
            exp_tokens = self._convert(exp)
            return base_tokens + exp_tokens + ['^']
        
        # 8. Functions (sin, cos, exp, ln, sqrt, abs, etc.)
        if isinstance(expr, sp.Function):
            func_name = type(expr).__name__
            
            # Map to paper's function names
            mapped_name = self.config.FUNCTION_MAP.get(func_name, func_name.lower())
            
            # Convert argument
            if len(expr.args) == 1:
                arg_tokens = self._convert(expr.args[0])
                return arg_tokens + [mapped_name]
            else:
                # Multi-argument function - convert each arg
                result = []
                for arg in expr.args:
                    result.extend(self._convert(arg))
                result.append(mapped_name)
                return result
        
        # 9. Derivative (should not appear in final expressions, but handle gracefully)
        if isinstance(expr, sp.Derivative):
            # Evaluate the derivative
            evaluated = expr.doit()
            return self._convert(evaluated)
        
        # 10. Negative numbers (treat as multiplication by -1)
        if expr.is_Mul and expr.args[0] == -1:
            # This is -something
            inner = sp.Mul(*expr.args[1:])
            inner_tokens = self._convert(inner)
            return inner_tokens + ['neg']  # Or use const and *
        
        # 11. Handle expressions like -x (negative variable)
        if expr.is_Mul and len(expr.args) == 2 and expr.args[0] == -1:
            inner_tokens = self._convert(expr.args[1])
            return inner_tokens + ['neg']
        
        # Fallback: try to evaluate and convert
        try:
            # Attempt to simplify and retry
            simplified = sp.simplify(expr)
            if simplified != expr:
                return self._convert(simplified)
        except:
            pass
        
        raise ValueError(f"Cannot convert expression to RPN: {type(expr)} -> {expr}")
    
    def to_rpn_string(self, expr: sp.Expr) -> str:
        """Convert expression to space-separated RPN string."""
        tokens = self.to_rpn(expr)
        return ' '.join(tokens)


def extract_operators(rpn_tokens: List[str]) -> List[str]:
    """
    Extract unique operators from RPN token list.
    
    Paper Section 3.4:
    "The first step involves identifying the unique set of operators by 
    removing duplicate tokens from the sequence."
    
    Args:
        rpn_tokens: List of RPN tokens
        
    Returns:
        List of unique operator tokens
    """
    # Operators are: +, -, *, /, sin, cos, tan, exp, ln, sqrt, abs, ^2, ^3, ^4, neg
    operators = {'+', '-', '*', '/', 'sin', 'cos', 'tan', 'exp', 'ln', 
                 'sqrt', 'abs', '^2', '^3', '^4', '^', 'neg'}
    
    # Also include variable tokens as potential operators
    # (they represent the "input" to the expression)
    unique_ops = []
    seen = set()
    
    for token in rpn_tokens:
        if token in operators or token.startswith('x'):
            if token not in seen:
                unique_ops.append(token)
                seen.add(token)
    
    return unique_ops


def encode_as_binary_vector(
    operators: List[str],
    operator_dict: List[str]
) -> List[int]:
    """
    Encode operator set as binary vector for comparison.
    
    Paper Section 3.4 and Table 1:
    "We encode each operator set as a binary vector over a fixed dictionary 
    of n possible operators"
    
    Args:
        operators: List of operators present
        operator_dict: Fixed dictionary of all possible operators
        
    Returns:
        Binary vector (1 if operator present, 0 otherwise)
    """
    op_set = set(operators)
    return [1 if op in op_set else 0 for op in operator_dict]


def assemble_data_point(
    pde_type: str,
    bc_type: str,
    boundary_face,  # BoundaryFace object
    f_expr: sp.Expr,
    g_expr: sp.Expr,
    u_expr: sp.Expr,
    include_solution: bool = True,
    g_flux_expr: Optional[sp.Expr] = None  # For Cauchy: second component
) -> Dict[str, Any]:
    """
    Assemble a complete training data point in the paper's format.
    
    Paper Figure 3 format:
    "Type: <PDE> | RHS: <ops> | <BC_Type>: <boundary> <ops> | Solution: || <ops>"
    
    For Cauchy boundary conditions:
    "Type: <PDE> | RHS: <ops> | Cauchy: <boundary> <g_value_ops> ||| <g_flux_ops> | Solution: || <ops>"
    The two Cauchy components are separated by '|||' to keep them distinct.
    
    Args:
        pde_type: PDE type string ('Poisson' or 'LinearConservationLaw')
        bc_type: Boundary condition type string
        boundary_face: BoundaryFace object
        f_expr: Right-hand side expression
        g_expr: Boundary condition expression (for Dirichlet/Neumann: the single g;
                for Cauchy: the g_value component)
        u_expr: Solution expression
        include_solution: Whether to include solution in output
        g_flux_expr: For Cauchy only - the normal derivative component (∂u/∂n)
        
    Returns:
        Dictionary with formatted data point
    """
    converter = RPNConverter()
    
    # Convert expressions to RPN
    f_rpn = converter.to_rpn(f_expr)
    g_rpn = converter.to_rpn(g_expr)
    u_rpn = converter.to_rpn(u_expr)
    
    # Format boundary face string (e.g., "x0=0")
    boundary_str = str(boundary_face)
    
    # Handle Cauchy separately - two components
    if bc_type == 'Cauchy' and g_flux_expr is not None:
        g_flux_rpn = converter.to_rpn(g_flux_expr)
        
        # Cauchy format: "Cauchy: <boundary> <g_value> ||| <g_flux>"
        # The ||| separator distinguishes the two independent constraints
        bc_part = f"Cauchy: {boundary_str} {' '.join(g_rpn)} ||| {' '.join(g_flux_rpn)}"
        
        input_parts = [
            f"Type: {pde_type}",
            f"RHS: {' '.join(f_rpn)}",
            bc_part,
            "Solution: ||"
        ]
        
        full_parts = input_parts[:-1] + [f"Solution: || {' '.join(u_rpn)}"]
        
        return {
            # Raw expressions for verification
            'u_expr': str(u_expr),
            'f_expr': str(f_expr),
            'g_value_expr': str(g_expr),  # Function value component
            'g_flux_expr': str(g_flux_expr),  # Normal derivative component
            
            # RPN tokens
            'u_rpn': u_rpn,
            'f_rpn': f_rpn,
            'g_value_rpn': g_rpn,
            'g_flux_rpn': g_flux_rpn,
            
            # LLM training format
            'input': " | ".join(input_parts),
            'target': ' '.join(u_rpn),
            'full': " | ".join(full_parts),
            
            # Metadata
            'pde_type': pde_type,
            'boundary_type': bc_type,
            'boundary_face': boundary_str
        }
    
    # Standard Dirichlet/Neumann case
    input_parts = [
        f"Type: {pde_type}",
        f"RHS: {' '.join(f_rpn)}",
        f"{bc_type}: {boundary_str} {' '.join(g_rpn)}",
        "Solution: ||"
    ]
    input_str = " | ".join(input_parts)
    
    # Construct target string (what the LLM should predict)
    target_str = ' '.join(u_rpn)
    
    # Full string with solution (for complete data point)
    full_parts = input_parts[:-1] + [f"Solution: || {' '.join(u_rpn)}"]
    full_str = " | ".join(full_parts)
    
    return {
        # Raw expressions for verification
        'u_expr': str(u_expr),
        'f_expr': str(f_expr),
        'g_expr': str(g_expr),
        
        # RPN tokens
        'u_rpn': u_rpn,
        'f_rpn': f_rpn,
        'g_rpn': g_rpn,
        
        # LLM training format
        'input': input_str,
        'target': target_str,
        'full': full_str,
        
        # Metadata
        'pde_type': pde_type,
        'boundary_type': bc_type,
        'boundary_face': boundary_str
    }


def assemble_batch(
    data_points: List[Tuple[str, str, Any, sp.Expr, sp.Expr, sp.Expr]]
) -> List[Dict[str, Any]]:
    """
    Assemble a batch of data points.
    
    Args:
        data_points: List of (pde_type, bc_type, boundary_face, f_expr, g_expr, u_expr)
        
    Returns:
        List of assembled data point dictionaries
    """
    results = []
    for pde_type, bc_type, boundary_face, f_expr, g_expr, u_expr in data_points:
        dp = assemble_data_point(pde_type, bc_type, boundary_face, f_expr, g_expr, u_expr)
        results.append(dp)
    return results


# Test function
def test_converter():
    """Test the RPN converter with sample expressions."""
    import sympy as sp
    
    x0, x1 = sp.symbols('x0 x1', real=True)
    converter = RPNConverter()
    
    print("Testing RPNConverter...")
    print("=" * 60)
    
    # Test 1: Simple polynomial
    expr1 = x0**2 + x1**2
    rpn1 = converter.to_rpn(expr1)
    print(f"Expression: {expr1}")
    print(f"RPN: {' '.join(rpn1)}")
    print()
    
    # Test 2: Trigonometric
    expr2 = sp.sin(x0) + sp.cos(x1)
    rpn2 = converter.to_rpn(expr2)
    print(f"Expression: {expr2}")
    print(f"RPN: {' '.join(rpn2)}")
    print()
    
    # Test 3: Product with constant
    expr3 = 2 * x0 * x1
    rpn3 = converter.to_rpn(expr3)
    print(f"Expression: {expr3}")
    print(f"RPN: {' '.join(rpn3)}")
    print()
    
    # Test 4: Exponential
    expr4 = sp.exp(x0) * sp.sin(x1)
    rpn4 = converter.to_rpn(expr4)
    print(f"Expression: {expr4}")
    print(f"RPN: {' '.join(rpn4)}")
    print()
    
    # Test 5: Mixed expression (matching paper example)
    expr5 = x0**2 * 2  # Should give "x0 ^2 const *"
    rpn5 = converter.to_rpn(expr5)
    print(f"Expression: {expr5}")
    print(f"RPN: {' '.join(rpn5)}")
    print()
    
    # Test full data point assembly
    print("=" * 60)
    print("Testing full data point assembly:")
    
    # Simulate a simple PDE case
    u = x0**2 + x1  # u = x0² + x1
    f = -sp.diff(u, x0, 2) - sp.diff(u, x1, 2)  # Poisson: f = -Δu
    
    from PDEDeriver import BoundaryFace
    boundary = BoundaryFace(x0, 0, -1)
    g = u.subs(x0, 0)  # Dirichlet at x0=0
    
    dp = assemble_data_point('Poisson', 'Dirichlet', boundary, f, g, u)
    print(f"\nInput: {dp['input']}")
    print(f"Target: {dp['target']}")
    print(f"\nFull: {dp['full']}")


if __name__ == "__main__":
    test_converter()
