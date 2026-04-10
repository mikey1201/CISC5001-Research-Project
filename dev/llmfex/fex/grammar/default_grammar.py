"""
Default FEX grammar specification.

This module defines the original FEX operator library as used in the
Finite Expression Method paper. This serves as the default grammar
when no LLM grammar is specified.
"""

from typing import List
from .grammar_spec import GrammarSpec, OperatorSpec

# Optional torch import - the implementations will be set up at module load
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def _get_numpy_compat_functions():
    """
    Get numpy/math compatible functions when torch is not available.
    These are used for testing and validation without requiring torch.
    """
    import math
    
    def np_exp(x):
        if hasattr(x, 'exp'):
            return x.exp()  # If it's a numpy-like array
        return math.exp(x) if isinstance(x, (int, float)) else [math.exp(v) for v in x]
    
    def np_sin(x):
        if hasattr(x, 'sin'):
            return x.sin()
        return math.sin(x) if isinstance(x, (int, float)) else [math.sin(v) for v in x]
    
    def np_cos(x):
        if hasattr(x, 'cos'):
            return x.cos()
        return math.cos(x) if isinstance(x, (int, float)) else [math.cos(v) for v in x]
    
    def np_abs(x):
        if hasattr(x, 'abs'):
            return x.abs()
        return abs(x) if isinstance(x, (int, float)) else [abs(v) for v in x]
    
    def np_sqrt(x):
        if hasattr(x, 'sqrt'):
            return x.sqrt()
        return math.sqrt(x) if isinstance(x, (int, float)) else [math.sqrt(v) for v in x]
    
    def np_tanh(x):
        if hasattr(x, 'tanh'):
            return x.tanh()
        return math.tanh(x) if isinstance(x, (int, float)) else [math.tanh(v) for v in x]
    
    def np_sigmoid(x):
        if hasattr(x, 'sigmoid'):
            return x.sigmoid()
        return 1.0 / (1.0 + math.exp(-x)) if isinstance(x, (int, float)) else [1.0/(1.0+math.exp(-v)) for v in x]
    
    def np_log(x):
        if hasattr(x, 'log'):
            return x.log()
        return math.log(x) if isinstance(x, (int, float)) else [math.log(v) for v in x]
    
    def np_maximum(x, y):
        return max(x, y) if isinstance(x, (int, float)) else [max(a, b) for a, b in zip(x, y)]
    
    def np_minimum(x, y):
        return min(x, y) if isinstance(x, (int, float)) else [min(a, b) for a, b in zip(x, y)]
    
    return {
        'exp': np_exp,
        'sin': np_sin,
        'cos': np_cos,
        'abs': np_abs,
        'sqrt': np_sqrt,
        'tanh': np_tanh,
        'sigmoid': np_sigmoid,
        'log': np_log,
        'maximum': np_maximum,
        'minimum': np_minimum,
    }


# Get implementations based on torch availability
if HAS_TORCH:
    _exp_impl = torch.exp
    _sin_impl = torch.sin
    _cos_impl = torch.cos
    _abs_impl = torch.abs
    _sqrt_impl = torch.sqrt
    _tanh_impl = torch.tanh
    _sigmoid_impl = torch.sigmoid
    _log_impl = torch.log
    _maximum_impl = lambda x, y: torch.maximum(x, y)
    _minimum_impl = lambda x, y: torch.minimum(x, y)
else:
    _np_funcs = _get_numpy_compat_functions()
    _exp_impl = _np_funcs['exp']
    _sin_impl = _np_funcs['sin']
    _cos_impl = _np_funcs['cos']
    _abs_impl = _np_funcs['abs']
    _sqrt_impl = _np_funcs['sqrt']
    _tanh_impl = _np_funcs['tanh']
    _sigmoid_impl = _np_funcs['sigmoid']
    _log_impl = _np_funcs['log']
    _maximum_impl = _np_funcs['maximum']
    _minimum_impl = _np_funcs['minimum']


# =============================================================================
# Default FEX Unary Operators
# =============================================================================

DEFAULT_UNARY_OPERATORS: List[OperatorSpec] = [
    # Identity/constant operators
    OperatorSpec(
        name="zero",
        arity="unary",
        implementation=lambda x: 0 * x**2,
        string_template="({}*(0)+{})",
        leaf_string_template="(0)",
        description="Returns zero (constant function)",
        token_aliases=["zero", "0", "const0"],
    ),
    OperatorSpec(
        name="one",
        arity="unary",
        implementation=lambda x: 1 + 0 * x**2,
        string_template="({}*(1)+{})",
        leaf_string_template="(1)",
        description="Returns one (constant function)",
        token_aliases=["one", "1", "const1"],
    ),
    
    # Linear/power operators
    OperatorSpec(
        name="identity",
        arity="unary",
        implementation=lambda x: x + 0 * x**2,
        string_template="({}*{}+{})",
        leaf_string_template="({})",
        description="Identity function f(x) = x",
        token_aliases=["identity", "id", "x", "linear"],
    ),
    OperatorSpec(
        name="square",
        arity="unary",
        implementation=lambda x: x**2,
        string_template="({}*({})**2+{})",
        leaf_string_template="(({})**2)",
        description="Square function f(x) = x^2",
        token_aliases=["square", "x2", "pow2", "**2", "^2"],
    ),
    OperatorSpec(
        name="cube",
        arity="unary",
        implementation=lambda x: x**3,
        string_template="({}*({})**3+{})",
        leaf_string_template="(({})**3)",
        description="Cube function f(x) = x^3",
        token_aliases=["cube", "x3", "pow3", "**3", "^3"],
    ),
    OperatorSpec(
        name="quad",
        arity="unary",
        implementation=lambda x: x**4,
        string_template="({}*({})**4+{})",
        leaf_string_template="(({})**4)",
        description="Quartic function f(x) = x^4",
        token_aliases=["quad", "x4", "pow4", "**4", "^4"],
    ),
    
    # Transcendental operators
    OperatorSpec(
        name="exp",
        arity="unary",
        implementation=_exp_impl,
        string_template="({}*exp({})+{})",
        leaf_string_template="(exp({}))",
        description="Exponential function",
        token_aliases=["exp", "exponential"],
    ),
    OperatorSpec(
        name="sin",
        arity="unary",
        implementation=_sin_impl,
        string_template="({}*sin({})+{})",
        leaf_string_template="(sin({}))",
        description="Sine function",
        token_aliases=["sin", "sine"],
    ),
    OperatorSpec(
        name="cos",
        arity="unary",
        implementation=_cos_impl,
        string_template="({}*cos({})+{})",
        leaf_string_template="(cos({}))",
        description="Cosine function",
        token_aliases=["cos", "cosine"],
    ),
]


# =============================================================================
# Default FEX Binary Operators
# =============================================================================

DEFAULT_BINARY_OPERATORS: List[OperatorSpec] = [
    OperatorSpec(
        name="add",
        arity="binary",
        implementation=lambda x, y: x + y,
        string_template="(({})+({}))",
        description="Addition",
        token_aliases=["add", "+", "plus", "sum"],
    ),
    OperatorSpec(
        name="mul",
        arity="binary",
        implementation=lambda x, y: x * y,
        string_template="(({})*({}))",
        description="Multiplication",
        token_aliases=["mul", "*", "times", "product"],
    ),
    OperatorSpec(
        name="sub",
        arity="binary",
        implementation=lambda x, y: x - y,
        string_template="(({})-({}))",
        description="Subtraction",
        token_aliases=["sub", "-", "minus"],
    ),
]


# =============================================================================
# Extended Operator Vocabulary (for LLM grammars)
# =============================================================================

EXTENDED_UNARY_OPERATORS: List[OperatorSpec] = [
    # Additional operators that may be used by LLM grammars
    OperatorSpec(
        name="neg",
        arity="unary",
        implementation=lambda x: -x,
        string_template="({}*(-({}))+{})",
        leaf_string_template="(-({}))",
        description="Negation",
        token_aliases=["neg", "negate", "-"],
    ),
    OperatorSpec(
        name="abs",
        arity="unary",
        implementation=_abs_impl,
        string_template="({}*abs({})+{})",
        leaf_string_template="(abs({}))",
        description="Absolute value",
        token_aliases=["abs", "absolute"],
    ),
    OperatorSpec(
        name="sqrt",
        arity="unary",
        implementation=_sqrt_impl,
        string_template="({}*sqrt({})+{})",
        leaf_string_template="(sqrt({}))",
        description="Square root",
        token_aliases=["sqrt", "squareroot"],
    ),
    OperatorSpec(
        name="tanh",
        arity="unary",
        implementation=_tanh_impl,
        string_template="({}*tanh({})+{})",
        leaf_string_template="(tanh({}))",
        description="Hyperbolic tangent",
        token_aliases=["tanh"],
    ),
    OperatorSpec(
        name="sigmoid",
        arity="unary",
        implementation=_sigmoid_impl,
        string_template="({}*sigmoid({})+{})",
        leaf_string_template="(sigmoid({}))",
        description="Sigmoid function",
        token_aliases=["sigmoid", "sig"],
    ),
    OperatorSpec(
        name="log",
        arity="unary",
        implementation=_log_impl,
        string_template="({}*log({})+{})",
        leaf_string_template="(log({}))",
        description="Natural logarithm",
        token_aliases=["log", "ln"],
    ),
    OperatorSpec(
        name="reciprocal",
        arity="unary",
        implementation=lambda x: 1.0 / x,
        string_template="({}*(1/{})+{})",
        leaf_string_template="(1/{})",
        description="Reciprocal 1/x",
        token_aliases=["reciprocal", "inv", "1/x"],
    ),
    # Additional power operators for LLM grammars
    # These allow LLM to predict exact powers like ^5, ^6, etc.
    OperatorSpec(
        name="pow5",
        arity="unary",
        implementation=lambda x: x**5,
        string_template="({}*({})**5+{})",
        leaf_string_template="(({})**5)",
        description="Fifth power function f(x) = x^5",
        token_aliases=["pow5", "^5", "**5", "x5"],
    ),
    OperatorSpec(
        name="pow6",
        arity="unary",
        implementation=lambda x: x**6,
        string_template="({}*({})**6+{})",
        leaf_string_template="(({})**6)",
        description="Sixth power function f(x) = x^6",
        token_aliases=["pow6", "^6", "**6", "x6"],
    ),
    # Additional power operators (pow7-pow10) for extended LLM support
    OperatorSpec(
        name="pow7",
        arity="unary",
        implementation=lambda x: x**7,
        string_template="({}*({})**7+{})",
        leaf_string_template="(({})**7)",
        description="Seventh power function f(x) = x^7",
        token_aliases=["pow7", "^7", "**7", "x7"],
    ),
    OperatorSpec(
        name="pow8",
        arity="unary",
        implementation=lambda x: x**8,
        string_template="({}*({})**8+{})",
        leaf_string_template="(({})**8)",
        description="Eighth power function f(x) = x^8",
        token_aliases=["pow8", "^8", "**8", "x8"],
    ),
    OperatorSpec(
        name="pow9",
        arity="unary",
        implementation=lambda x: x**9,
        string_template="({}*({})**9+{})",
        leaf_string_template="(({})**9)",
        description="Ninth power function f(x) = x^9",
        token_aliases=["pow9", "^9", "**9", "x9"],
    ),
    OperatorSpec(
        name="pow10",
        arity="unary",
        implementation=lambda x: x**10,
        string_template="({}*({})**10+{})",
        leaf_string_template="(({})**10)",
        description="Tenth power function f(x) = x^10",
        token_aliases=["pow10", "^10", "**10", "x10"],
    ),
]


EXTENDED_BINARY_OPERATORS: List[OperatorSpec] = [
    OperatorSpec(
        name="div",
        arity="binary",
        implementation=lambda x, y: x / y,
        string_template="(({})/({}))",
        description="Division",
        token_aliases=["div", "/", "divide"],
    ),
    OperatorSpec(
        name="max",
        arity="binary",
        implementation=_maximum_impl,
        string_template="(max({},{}))",
        description="Maximum",
        token_aliases=["max", "maximum"],
    ),
    OperatorSpec(
        name="min",
        arity="binary",
        implementation=_minimum_impl,
        string_template="(min({},{}))",
        description="Minimum",
        token_aliases=["min", "minimum"],
    ),
]


def get_default_fex_grammar() -> GrammarSpec:
    """
    Get the default FEX grammar specification.
    
    This returns the original FEX operator library as defined in the
    Finite Expression Method paper. This is the grammar used when
    no LLM grammar is specified.
    
    Returns:
        GrammarSpec with default FEX operators
    """
    return GrammarSpec(
        name="FEX_Default",
        unary_operators=DEFAULT_UNARY_OPERATORS.copy(),
        binary_operators=DEFAULT_BINARY_OPERATORS.copy(),
        metadata={
            "source": "FEX_paper",
            "version": "1.0",
            "description": "Default FEX operator library",
        },
    )


def get_extended_grammar() -> GrammarSpec:
    """
    Get an extended grammar with additional operators.
    
    This includes all default FEX operators plus additional operators
    that may be useful for certain PDE types or LLM-generated grammars.
    
    Returns:
        GrammarSpec with extended operator set
    """
    return GrammarSpec(
        name="FEX_Extended",
        unary_operators=DEFAULT_UNARY_OPERATORS + EXTENDED_UNARY_OPERATORS,
        binary_operators=DEFAULT_BINARY_OPERATORS + EXTENDED_BINARY_OPERATORS,
        metadata={
            "source": "FEX_extended",
            "version": "1.0",
            "description": "Extended FEX operator library",
        },
    )


def get_operator_vocabulary() -> dict:
    """
    Get a dictionary mapping operator names to their specifications.
    
    This is useful for LLM grammar adapters to look up operators
    from token names.
    
    Returns:
        Dictionary mapping operator name (lowercase) to OperatorSpec
    """
    vocabulary = {}
    
    for op in DEFAULT_UNARY_OPERATORS + EXTENDED_UNARY_OPERATORS:
        for alias in op.token_aliases:
            vocabulary[alias.lower()] = op
    
    for op in DEFAULT_BINARY_OPERATORS + EXTENDED_BINARY_OPERATORS:
        for alias in op.token_aliases:
            vocabulary[alias.lower()] = op
    
    return vocabulary
