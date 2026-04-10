"""
Problem Configuration for FEX.

This module allows FEX to solve different PDE problems by loading
problem-specific RHS and boundary conditions from configuration files.

The configuration is loaded from a JSON file that specifies:
- RHS function in postfix notation
- Boundary conditions
- True solution (for evaluation)

Usage:
    # Set the problem config file before importing function.py
    import problem_config
    problem_config.load_config('problem_0_config.json')
    
    # Then import function which will use the loaded config
    import function
"""

import json
import os
import torch
import math
from typing import Dict, Any, Optional, Callable, List

# Global config state
_current_config: Optional[Dict[str, Any]] = None
_config_path: Optional[str] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a problem configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        The loaded configuration dictionary
    """
    global _current_config, _config_path
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        _current_config = json.load(f)
    
    _config_path = config_path
    print(f"Loaded problem config from: {config_path}")
    print(f"  Problem ID: {_current_config.get('problem_id', 'unknown')}")
    print(f"  PDE Type: {_current_config.get('pde_type', 'unknown')}")
    
    return _current_config


def get_config() -> Optional[Dict[str, Any]]:
    """Get the currently loaded configuration."""
    return _current_config


def set_config(config: Dict[str, Any]) -> None:
    """Set the configuration directly (without loading from file)."""
    global _current_config
    _current_config = config


def reset_config() -> None:
    """Reset to default (no config loaded)."""
    global _current_config, _config_path
    _current_config = None
    _config_path = None


# =============================================================================
# Postfix Expression Evaluator
# =============================================================================

def evaluate_postfix(tokens: List[str], variables: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a postfix expression with given variables.
    
    Args:
        tokens: List of tokens in postfix notation
        variables: Tensor of shape (batch_size, dim) with variable values
                   Variables are x0, x1, x2, ... (0-indexed for FEX)
        
    Returns:
        Tensor of shape (batch_size, 1) with evaluated values
    """
    bs = variables.size(0)
    dim = variables.size(1)
    
    # Split tokens if they're a single string
    if isinstance(tokens, str):
        tokens = tokens.strip().split()
    
    stack = []
    
    for token in tokens:
        token = token.strip()
        
        # Variable (x1, x2, x3, ... - convert to 0-indexed)
        if token.startswith('x') and token[1:].isdigit():
            var_idx = int(token[1:]) - 1  # Convert 1-indexed to 0-indexed
            if var_idx < dim:
                stack.append(variables[:, var_idx:var_idx+1])
            else:
                stack.append(torch.zeros(bs, 1).cuda())
        
        # Constant (const or numeric)
        elif token == 'const' or token == 'c':
            # Constant will be learned by the network
            stack.append(torch.ones(bs, 1).cuda())
        
        elif token.replace('.', '').replace('-', '').isdigit():
            stack.append(float(token) * torch.ones(bs, 1).cuda())
        
        # Binary operators
        elif token in ['*', 'mul']:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif token in ['+', 'add']:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif token in ['-', 'sub']:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif token in ['/', 'div']:
            b, a = stack.pop(), stack.pop()
            stack.append(a / (b + 1e-10))
        
        # Unary operators
        elif token in ['exp']:
            a = stack.pop()
            stack.append(torch.exp(a))
        elif token in ['sin']:
            a = stack.pop()
            stack.append(torch.sin(a))
        elif token in ['cos']:
            a = stack.pop()
            stack.append(torch.cos(a))
        elif token in ['log', 'ln']:
            a = stack.pop()
            stack.append(torch.log(torch.abs(a) + 1e-10))
        elif token in ['sqrt']:
            a = stack.pop()
            stack.append(torch.sqrt(torch.abs(a) + 1e-10))
        elif token in ['abs']:
            a = stack.pop()
            stack.append(torch.abs(a))
        
        # Power operators
        elif token.startswith('^') and token[1:].isdigit():
            power = int(token[1:])
            a = stack.pop()
            stack.append(torch.pow(a, power))
        
        else:
            # Unknown token - treat as constant
            stack.append(torch.ones(bs, 1).cuda())
    
    if len(stack) == 0:
        return torch.zeros(bs, 1).cuda()
    
    result = stack[-1]
    if result.size(1) != 1:
        result = result.mean(dim=1, keepdim=True)
    
    return result


# =============================================================================
# RHS Function
# =============================================================================

def RHS_pde_configured(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the RHS of the PDE based on loaded configuration.
    
    If no config is loaded, uses default (-dim).
    """
    global _current_config
    
    bs = x.size(0)
    dim = x.size(1)
    
    if _current_config is None:
        # Default: -dim (constant RHS for standard Poisson)
        return -dim * torch.ones(bs, 1).cuda()
    
    rhs_expr = _current_config.get('rhs', None)
    
    if rhs_expr is None:
        return -dim * torch.ones(bs, 1).cuda()
    
    # Evaluate RHS expression
    try:
        result = evaluate_postfix(rhs_expr, x)
        return -result  # Negate because FEX solves -∇²u = f
    except Exception as e:
        print(f"Warning: Error evaluating RHS: {e}, using default")
        return -dim * torch.ones(bs, 1).cuda()


# =============================================================================
# True Solution Function
# =============================================================================

def true_solution_configured(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the true solution based on loaded configuration.
    
    If no config is loaded, uses default (0.5*sum(x²)).
    """
    global _current_config
    
    bs = x.size(0)
    dim = x.size(1)
    
    if _current_config is None:
        # Default: 0.5 * sum(x²)
        return 0.5 * torch.sum(x**2, dim=1, keepdim=True)
    
    solution_expr = _current_config.get('solution', None)
    
    if solution_expr is None:
        return 0.5 * torch.sum(x**2, dim=1, keepdim=True)
    
    # Evaluate solution expression
    try:
        result = evaluate_postfix(solution_expr, x)
        return result
    except Exception as e:
        print(f"Warning: Error evaluating solution: {e}, using default")
        return 0.5 * torch.sum(x**2, dim=1, keepdim=True)


# =============================================================================
# Boundary Conditions
# =============================================================================

def get_boundary_values(bd_pts: torch.Tensor) -> torch.Tensor:
    """
    Get boundary condition values at given points.
    
    If no config is loaded, uses the true solution.
    """
    global _current_config
    
    if _current_config is None:
        return true_solution_configured(bd_pts)
    
    boundary_conditions = _current_config.get('boundary_conditions', [])
    
    if not boundary_conditions:
        return true_solution_configured(bd_pts)
    
    # For now, use the true solution at boundary
    # (Full boundary condition support would require more complex handling)
    return true_solution_configured(bd_pts)


# =============================================================================
# Config File Generation
# =============================================================================

def create_problem_config(
    problem_id: int,
    pde_type: str,
    rhs: str,
    solution: str,
    boundary_conditions: List[Dict],
    output_path: str,
) -> Dict[str, Any]:
    """
    Create a problem configuration file.
    
    Args:
        problem_id: Problem identifier
        pde_type: Type of PDE (Poisson, LinearConservation, etc.)
        rhs: RHS expression in postfix notation
        solution: True solution in postfix notation
        boundary_conditions: List of boundary condition specs
        output_path: Path to save the config file
        
    Returns:
        The created configuration dictionary
    """
    config = {
        "problem_id": problem_id,
        "pde_type": pde_type,
        "rhs": rhs,
        "solution": solution,
        "boundary_conditions": boundary_conditions,
        "variables": ["x1", "x2", "x3"],  # Default 3D
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Problem config utilities')
    parser.add_argument('--config', type=str, help='Path to config file to load')
    parser.add_argument('--test', action='store_true', help='Run test')
    
    args = parser.parse_args()
    
    if args.test:
        # Test with a sample problem
        test_config = {
            "problem_id": 0,
            "pde_type": "Poisson",
            "rhs": "const x2 * x3 exp *",
            "solution": "const x2 * x3 exp *",
        }
        set_config(test_config)
        
        # Test evaluation
        x = torch.randn(10, 3).cuda()
        print(f"Input shape: {x.shape}")
        print(f"RHS: {RHS_pde_configured(x)[:5]}")
        print(f"Solution: {true_solution_configured(x)[:5]}")
    
    elif args.config:
        load_config(args.config)
        print(f"Loaded config: {get_config()}")