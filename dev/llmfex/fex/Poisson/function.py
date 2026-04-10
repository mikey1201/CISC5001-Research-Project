"""
FEX Function Module with Problem Configuration Support.

This module provides the PDE functions for FEX. It now supports loading
problem-specific configurations via the problem_config module.

If a problem config is loaded (via problem_config.load_config()), the
RHS and true solution will be read from the config. Otherwise, defaults
are used.
"""

import numpy as np
import torch
from torch import sin, cos, exp
import math

# Import problem configuration support
import problem_config

def LHS_pde(u, x, dim_set):
    """
    Compute the left-hand side of the PDE: -∇²u
    
    This is the Laplacian with negative sign, used for Poisson-type equations.
    """
    v = torch.ones(u.shape).cuda()
    bs = x.size(0)
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    uxx = torch.zeros(bs, dim_set).cuda()
    for i in range(dim_set):
        ux_tem = ux[:, i:i+1]
        uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]
        uxx[:, i] = uxx_tem[:, i]
    LHS = -torch.sum(uxx, dim=1, keepdim=True)
    return LHS


def RHS_pde(x):
    """
    Compute the right-hand side of the PDE.
    
    Uses problem_config if loaded, otherwise uses default (-dim).
    """
    return problem_config.RHS_pde_configured(x)


def true_solution(x):
    """
    Compute the true/reference solution.
    
    Uses problem_config if loaded, otherwise uses default (0.5*sum(x²)).
    """
    return problem_config.true_solution_configured(x)


# =============================================================================
# Default Operator Library (used by original FEX)
# =============================================================================

unary_functions = [lambda x: 0*x**2,
                   lambda x: 1+0*x**2,
                   lambda x: x+0*x**2,
                   lambda x: x**2,
                   lambda x: x**3,
                   lambda x: x**4,
                   torch.exp,
                   torch.sin,
                   torch.cos,]

binary_functions = [lambda x,y: x+y,
                    lambda x,y: x*y,
                    lambda x,y: x-y]


unary_functions_str = ['({}*(0)+{})',
                       '({}*(1)+{})',
                       '({}*{}+{})',
                       '({}*({})**2+{})',
                       '({}*({})**3+{})',
                       '({}*({})**4+{})',
                       '({}*exp({})+{})',
                       '({}*sin({})+{})',
                       '({}*cos({})+{})',]

unary_functions_str_leaf= ['(0)',
                           '(1)',
                           '({})',
                           '(({})**2)',
                           '(({})**3)',
                           '(({})**4)',
                           '(exp({}))',
                           '(sin({}))',
                           '(cos({}))',]


binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))']


# =============================================================================
# Test/Validation
# =============================================================================

if __name__ == '__main__':
    batch_size = 200
    left = -1
    right = 1
    
    # Test with default config
    print("=" * 60)
    print("Testing with DEFAULT configuration")
    print("=" * 60)
    
    points = (torch.rand(batch_size, 1)) * (right - left) + left
    x = torch.autograd.Variable(points.cuda(), requires_grad=True)
    function = true_solution

    LHS = LHS_pde(function(x), x)
    RHS = RHS_pde(x)
    pde_loss = torch.nn.functional.mse_loss(LHS, RHS)

    bc_points = torch.FloatTensor([[left], [right]]).cuda()
    bc_value = true_solution(bc_points)
    bd_loss = torch.nn.functional.mse_loss(function(bc_points), bc_value)

    print(f'PDE loss: {pde_loss.item():.6f} -- Boundary loss: {bd_loss.item():.6f}')
    
    # Test with custom config
    print("\n" + "=" * 60)
    print("Testing with CUSTOM configuration")
    print("=" * 60)
    
    test_config = {
        "problem_id": 0,
        "pde_type": "Poisson",
        "rhs": "const",  # Constant RHS
        "solution": "const x1 *",  # u = c * x
    }
    problem_config.set_config(test_config)
    
    # For 1D problem
    x2 = torch.autograd.Variable(points.cuda(), requires_grad=True)
    LHS2 = LHS_pde(true_solution(x2), x2)
    RHS2 = RHS_pde(x2)
    pde_loss2 = torch.nn.functional.mse_loss(LHS2, RHS2)
    print(f'Custom config PDE loss: {pde_loss2.item():.6f}')
    
    # Reset to default
    problem_config.reset_config()
    print("\nReset to default configuration")
