# PDE Data Generation for LLM Training

This package generates training data for fine-tuning Large Language Models (LLMs) to predict symbolic operators in Partial Differential Equation (PDE) solutions.

## Paper Reference

Based on methodology from:
> **"From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs"**
> 
> Rohan Bhatnagar, Ling Liang, Krish Patel, Haizhao Yang
> 
> University of Maryland & University of Tennessee

## Key Refinements Made

### 1. Operator Sets (Paper Sections 3.1 & 5.1)

**Binary Operators:**
- Reduced set (matching paper): `+`, `-`, `*`
- Full set: `+`, `-`, `*`, `/`

**Unary Operators:**
- Reduced set (matching paper): `sin`, `cos`, `exp`, `pow2`, `pow3`, `pow4`, `id`
- Full set: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `abs`, `pow2`, `pow3`, `pow4`, `neg`

### 2. α, β Parameterization (Paper Section 3.1)

Each unary operator can be composed with affine transformation:
```
α·u(·) + β
```
where `u(·)` is the unary operation and `α`, `β` are scalar parameters.

**Implementation:**
- `α` range: [-4, 4] (non-zero)
- `β` range: [-4, 4]
- Applied probabilistically (50% chance) to avoid overly complex expressions

### 3. Binary Computational Trees (Paper Section 3.1)

- Maximum depth: 3 (as specified in paper)
- Node types: leaf (variable or constant), unary operator, binary operator
- Probability distribution at root: 5% leaf, 40% unary, 55% binary
- Ensures final expression contains at least one variable

### 4. PDE Types (Paper Section 5.1)

**Poisson Equation:**
```
-Δu = f  in Ω
u = g     on ∂Ω
```
where `f = -Σ ∂²u/∂xᵢ²`

**Linear Conservation Law:**
```
∇·u = f   in Ω
```
where `f = Σ ∂u/∂xᵢ`

### 5. Boundary Conditions (Paper Section 5.1)

- **Dirichlet:** `g = u|∂Ω`
- **Neumann:** `g = ∂u/∂n|∂Ω`
- **Cauchy:** `g = u + ∂u/∂n` (combined condition)

### 6. RPN Format (Paper Figure 3)

Output format:
```
Type: <PDE> | RHS: <f_ops> | <BC_Type>: <boundary> <g_ops> | Solution: || <u_ops>
```

Example:
```
Type: Poisson | RHS: const x0 * | Dirichlet: x0=0 const | Solution: || x0 ^2 const *
```

**Token conventions:**
- Constants: `const`
- Variables: `x0`, `x1`, `x2`, ...
- Powers: `^2`, `^3`, `^4`, ...
- Functions: `sin`, `cos`, `exp`, `ln`, `sqrt`, `abs`

### 7. Quality Controls

1. **Domain Validation:** Expressions are tested at random points in domain [0.1, 2.0]
2. **Complexity Check:** 
   - Max expression string length: 500 characters
   - Max nested operations: 8
   - Max power: 4 (rejects `**5`, `**6`, etc.)
3. **Variable Requirement:** Final expression must contain at least one variable

## Usage

### Command Line

```bash
# Generate 1000 samples per PDE type
python DGen.py --samples 1000

# Generate full 198K dataset (as in paper)
python DGen.py --full

# Custom configuration
python DGen.py --samples 500 --depth 3 --vars 2 --output ./my_data --seed 42
```

### As Python Module

```python
from data_generation import (
    RandomExpressionGenerator,
    PDEDeriver,
    assemble_data_point,
    create_boundary_face
)

# Initialize
gen = RandomExpressionGenerator(num_vars=2, max_depth=3)
deriver = PDEDeriver(gen.vars)

# Generate expression
u_expr, metadata = gen.generate_u()

# Compute PDE components
f_expr = deriver.calculate_f(u_expr, 'Poisson')
boundary = create_boundary_face(gen.vars, value=0.0)
g_expr = deriver.calculate_g(u_expr, 'Dirichlet', boundary)

# Assemble data point
data_point = assemble_data_point(
    'Poisson', 'Dirichlet', boundary, f_expr, g_expr, u_expr
)
```

## Output Structure

Each data point contains:

```json
{
    "u_expr": "sin(x0*x1) - cos(x0*x1) - 1",
    "f_expr": "x0**2*sin(x0*x1) - x0**2*cos(x0*x1) + ...",
    "g_expr": "x1*sin(1.0*x1) + x1*cos(1.0*x1) + ...",
    "u_rpn": ["const", "const", "x0", "x1", "*", "cos", "*", "+", ...],
    "f_rpn": ["x0", "^2", "x0", "x1", "*", "sin", "*", ...],
    "g_rpn": ["const", "const", "const", "x1", "*", "cos", "*", "+", ...],
    "input": "Type: Poisson | RHS: x0 ^2 ... | Cauchy: x0=1.0 ... | Solution: ||",
    "target": "const const x0 x1 * cos * + ...",
    "full": "Type: Poisson | RHS: ... | Solution: || const const ...",
    "pde_type": "Poisson",
    "boundary_type": "Cauchy",
    "boundary_face": "x0=1.0"
}
```

## File Structure

```
data_generation/
├── __init__.py     # Package initialization
├── PDE.py          # PDE type specifications
├── REGen.py        # Random expression generator
├── Deriver.py      # PDE derivation (f and g calculation)
├── RPNC.py         # RPN conversion and data assembly
├── DGen.py         # Main generation pipeline
└── README_DATA.md  # This documentation
```

## Statistics (Paper Section 5.1)

- **Total samples:** 198,000
- **Per PDE type:** 99,000
- **Per combination:** 33,000 (PDE type × BC type)
- **Tree depth:** 3
- **Variables:** x0, x1, x2 (default)

## LLM Training (Paper Figure 3)

The generated data is designed for fine-tuning decoder-only LLMs (like LLaMA):

1. **Input:** The `input` field (everything before `||`)
2. **Target:** The `target` field (everything after `||`)
3. **Loss:** Cross-entropy on token prediction

Example training prompt:
```
Type: Poisson | RHS: const x0 * | Dirichlet: x0=0 const | Solution: ||
```
Expected completion:
```
x0 ^2 const *
```
