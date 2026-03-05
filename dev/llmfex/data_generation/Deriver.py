"""
Deriver.py - Refined PDE Derivation Module

Derives the right-hand side function f and boundary condition g from a given
solution u(x), following the paper's mathematical formulation.

Paper Section 1, Equation (1.1):
    Du = f, in Ω
    Bu = g, on ∂Ω

PDE Types (Section 5.1):
- Poisson: -Δu = f (negative Laplacian)
- LinearConservationLaw: ∇·u = f (divergence)

Boundary Conditions:
- Dirichlet: u|∂Ω = g
- Neumann: ∂u/∂n|∂Ω = g
- Cauchy: u + ∂u/∂n = g (combined condition)
"""

import sympy as sp
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BoundaryFace:
    """
    Represents a boundary face of the domain.
    
    For a hypercube domain [0,1]^d, boundary faces are axis-aligned:
    - (x0, 0): face where x0 = 0
    - (x0, 1): face where x0 = 1
    - etc.
    """
    variable: sp.Symbol
    value: float
    normal_direction: int  # +1 or -1 (outward normal direction)
    
    def __str__(self) -> str:
        return f"{self.variable}={self.value}"
    
    def get_normal_vector(self) -> Dict[sp.Symbol, int]:
        """
        Get the outward unit normal vector for this boundary face.
        
        For axis-aligned boundaries, normal is a unit vector in the 
        direction of the variable, with sign determined by face position.
        """
        return {self.variable: self.normal_direction}


class Deriver:
    """
    Derives PDE components (f and g) from a given solution u.
    
    Following paper's formulation in Section 1 and experimental setup in Section 5.1.
    """
    
    def __init__(self, vars_list: List[sp.Symbol]):
        """
        Initialize the deriver with the list of spatial variables.
        
        Args:
            vars_list: List of SymPy symbols representing spatial variables
        """
        self.vars = vars_list
        self.dim = len(vars_list)
    
    def calculate_f(self, u_expr: sp.Expr, pde_type: str) -> sp.Expr:
        """
        Calculate the right-hand side function f from solution u.
        
        Paper definitions:
        - Poisson (Section 2): -Δu = f, so f = -Δu = -Σ ∂²u/∂xᵢ²
        - LinearConservationLaw: ∇·u = Σ ∂u/∂xᵢ = f
        
        Args:
            u_expr: The solution expression u(x)
            pde_type: 'Poisson' or 'LinearConservationLaw'
            
        Returns:
            The right-hand side function f(x)
        """
        if pde_type == 'Poisson':
            # Paper Equation (2.1): -Δu = f
            # Laplacian: Δu = Σ ∂²u/∂xᵢ²
            laplacian = sum(sp.diff(u_expr, v, 2) for v in self.vars)
            f_expr = -laplacian
            
        elif pde_type == 'LinearConservationLaw':
            # Linear conservation law: ∇·u = Σ ∂u/∂xᵢ = f
            divergence = sum(sp.diff(u_expr, v, 1) for v in self.vars)
            f_expr = divergence
            
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}. "
                           f"Supported types: Poisson, LinearConservationLaw")
        
        return sp.expand(f_expr)
    
    def calculate_g(
        self,
        u_expr: sp.Expr,
        boundary_type: str,
        boundary_face: BoundaryFace
    ) -> sp.Expr:
        """
        Calculate the boundary condition function g.
        
        Paper Section 1 boundary operators:
        - Dirichlet: B = γ₀, g = u|∂Ω
        - Neumann: B = γ₁, g = ∂u/∂n|∂Ω
        - Cauchy: TWO separate constraints (u AND ∂u/∂n)
        
        Args:
            u_expr: The solution expression u(x)
            boundary_type: 'Dirichlet', 'Neumann', or 'Cauchy'
            boundary_face: The boundary face specification
            
        Returns:
            For Dirichlet/Neumann: single expression g
            For Cauchy: This method should not be used directly; use calculate_g_cauchy instead
        """
        var = boundary_face.variable
        val = boundary_face.value
        
        # Compute normal derivative: ∂u/∂n = n·∇u
        # For axis-aligned boundary, this is ±∂u/∂xᵢ
        normal_dir = boundary_face.normal_direction
        normal_derivative = normal_dir * sp.diff(u_expr, var, 1)
        
        if boundary_type == 'Dirichlet':
            # g = u|∂Ω
            g_expr = u_expr
            g_expr_evaluated = g_expr.subs(var, val)
            return sp.expand(g_expr_evaluated)
            
        elif boundary_type == 'Neumann':
            # g = ∂u/∂n|∂Ω
            g_expr = normal_derivative
            g_expr_evaluated = g_expr.subs(var, val)
            return sp.expand(g_expr_evaluated)
            
        elif boundary_type == 'Cauchy':
            # CAUTION: Cauchy requires TWO separate components
            # Do NOT sum them - use calculate_g_cauchy() instead
            raise ValueError(
                "Cauchy boundary conditions require two separate components. "
                "Use calculate_g_cauchy() instead of calculate_g()."
            )
            
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}. "
                           f"Supported types: Dirichlet, Neumann, Cauchy")
    
    def calculate_g_cauchy(
        self,
        u_expr: sp.Expr,
        boundary_face: BoundaryFace
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Calculate the TWO components of a Cauchy boundary condition.
        
        A Cauchy boundary condition specifies BOTH:
        - The function value: u|∂Ω = g_value
        - The normal derivative: ∂u/∂n|∂Ω = g_flux
        
        These are INDEPENDENT constraints and must be kept separate.
        Summing them destroys the information the LLM needs to learn.
        
        Args:
            u_expr: The solution expression u(x)
            boundary_face: The boundary face specification
            
        Returns:
            Tuple of (g_value, g_flux) - both evaluated at the boundary
        """
        var = boundary_face.variable
        val = boundary_face.value
        normal_dir = boundary_face.normal_direction
        
        # Component 1: Function value u|∂Ω
        g_value = u_expr.subs(var, val)
        g_value = sp.expand(g_value)
        
        # Component 2: Normal derivative ∂u/∂n|∂Ω
        normal_derivative = normal_dir * sp.diff(u_expr, var, 1)
        g_flux = normal_derivative.subs(var, val)
        g_flux = sp.expand(g_flux)
        
        return g_value, g_flux
    
    def calculate_all_boundary_conditions(
        self,
        u_expr: sp.Expr,
        boundary_type: str,
        domain_bounds: Tuple[float, float] = (0, 1)
    ) -> List[Tuple[BoundaryFace, sp.Expr]]:
        """
        Calculate boundary conditions on all faces of a hypercube domain.
        
        For a d-dimensional hypercube, there are 2d boundary faces.
        
        Args:
            u_expr: The solution expression
            boundary_type: Type of boundary condition
            domain_bounds: (lower, upper) bounds for all dimensions
            
        Returns:
            List of (boundary_face, g_expression) tuples
        """
        results = []
        lower, upper = domain_bounds
        
        for var in self.vars:
            # Lower boundary face (xᵢ = lower)
            face_lower = BoundaryFace(var, lower, -1)  # Normal points inward (negative direction)
            g_lower = self.calculate_g(u_expr, boundary_type, face_lower)
            results.append((face_lower, g_lower))
            
            # Upper boundary face (xᵢ = upper)  
            face_upper = BoundaryFace(var, upper, +1)  # Normal points outward (positive direction)
            g_upper = self.calculate_g(u_expr, boundary_type, face_upper)
            results.append((face_upper, g_upper))
        
        return results


def create_boundary_face(
    vars_list: List[sp.Symbol],
    var_idx: Optional[int] = None,
    value: float = 0.0
) -> BoundaryFace:
    """
    Create a boundary face specification.
    
    Args:
        vars_list: List of variables
        var_idx: Index of the variable for this boundary (random if None)
        value: Boundary value (typically 0 or 1)
        
    Returns:
        BoundaryFace object
    """
    import random
    
    if var_idx is None:
        var_idx = random.randint(0, len(vars_list) - 1)
    
    var = vars_list[var_idx]
    
    # Normal direction: +1 if value > 0.5, -1 otherwise
    normal_dir = +1 if value >= 0.5 else -1
    
    return BoundaryFace(var, value, normal_dir)


# Test function
def test_deriver():
    """Test the PDE deriver with sample expressions."""
    import sympy as sp
    
    # Create variables
    x0, x1 = sp.symbols('x0 x1', real=True)
    vars_list = [x0, x1]
    
    deriver = PDEDeriver(vars_list)
    
    # Test case 1: Simple polynomial
    u1 = x0**2 + x1**2
    
    print("=" * 60)
    print("Test 1: u = x0² + x1²")
    print("=" * 60)
    
    f_poisson = deriver.calculate_f(u1, 'Poisson')
    print(f"Poisson f = -Δu = {f_poisson}")
    
    f_conservation = deriver.calculate_f(u1, 'LinearConservationLaw')
    print(f"Conservation f = ∇·u = {f_conservation}")
    
    # Test boundary conditions
    boundary = create_boundary_face(vars_list, var_idx=0, value=0.0)
    print(f"\nBoundary face: {boundary}")
    
    g_dirichlet = deriver.calculate_g(u1, 'Dirichlet', boundary)
    print(f"Dirichlet g = {g_dirichlet}")
    
    g_neumann = deriver.calculate_g(u1, 'Neumann', boundary)
    print(f"Neumann g = {g_neumann}")
    
    g_cauchy = deriver.calculate_g(u1, 'Cauchy', boundary)
    print(f"Cauchy g = {g_cauchy}")
    
    # Test case 2: Expression with trig functions
    print("\n" + "=" * 60)
    print("Test 2: u = sin(x0) + cos(x1)")
    print("=" * 60)
    
    u2 = sp.sin(x0) + sp.cos(x1)
    
    f_poisson2 = deriver.calculate_f(u2, 'Poisson')
    print(f"Poisson f = -Δu = {f_poisson2}")
    
    g_dirichlet2 = deriver.calculate_g(u2, 'Dirichlet', boundary)
    print(f"Dirichlet g (x0=0) = {g_dirichlet2}")


if __name__ == "__main__":
    test_deriver()
