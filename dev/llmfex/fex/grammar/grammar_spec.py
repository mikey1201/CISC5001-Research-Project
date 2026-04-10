"""
Core grammar specification classes for FEX.

This module defines the data structures for representing grammars
(operator sets) used in the FEX expression search space.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple

# Optional torch import for type checking
try:
    import torch
except ImportError:
    torch = None  # Will be handled at runtime


@dataclass
class OperatorSpec:
    """
    Specification for a single operator in the FEX grammar.
    
    Attributes:
        name: Canonical name of the operator (e.g., "sin", "add", "exp")
        arity: Either "unary" or "binary"
        implementation: The actual callable function
        string_template: Template for string representation
            - For unary: format with (scaling, input, bias) -> "({}*sin({})+{})"
            - For binary: format with (left, right) -> "(({})+({}))"
        leaf_string_template: Optional template for leaf nodes (unary only)
        description: Optional human-readable description
        token_aliases: Alternative names this operator can be referenced by
            (e.g., ["add", "+", "plus"] for addition)
    """
    name: str
    arity: str  # "unary" or "binary"
    implementation: Callable
    string_template: str
    leaf_string_template: Optional[str] = None
    description: str = ""
    token_aliases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate operator specification after initialization."""
        if self.arity not in ("unary", "binary"):
            raise ValueError(f"Invalid arity '{self.arity}'. Must be 'unary' or 'binary'.")
        
        # Ensure name is in aliases
        if self.name not in self.token_aliases:
            self.token_aliases = [self.name] + self.token_aliases
    
    def matches_token(self, token: str) -> bool:
        """Check if a token matches this operator."""
        token_lower = token.lower().strip()
        return token_lower in [alias.lower() for alias in self.token_aliases]
    
    def get_leaf_template(self) -> str:
        """Get the appropriate string template for leaf nodes."""
        if self.leaf_string_template:
            return self.leaf_string_template
        # Default leaf template for unary operators
        if self.arity == "unary":
            return f"({self.name}({{}}))"
        return self.string_template


@dataclass
class GrammarSpec:
    """
    Complete grammar specification for FEX expression search.
    
    A grammar defines the set of operators available for constructing
    mathematical expressions in FEX. This includes both unary operators
    (sin, cos, exp, etc.) and binary operators (+, *, -, etc.).
    
    Attributes:
        unary_operators: List of unary operator specifications
        binary_operators: List of binary operator specifications
        metadata: Optional metadata about the grammar source
        name: Human-readable name for this grammar
    """
    unary_operators: List[OperatorSpec]
    binary_operators: List[OperatorSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)
    name: str = "unnamed_grammar"
    
    def get_unary_functions(self) -> List[Callable]:
        """Get list of unary operator implementations."""
        return [op.implementation for op in self.unary_operators]
    
    def get_binary_functions(self) -> List[Callable]:
        """Get list of binary operator implementations."""
        return [op.implementation for op in self.binary_operators]
    
    def get_unary_strings(self) -> List[str]:
        """
        Get string templates for unary operators (non-leaf nodes).
        
        Templates include scaling and bias parameters:
        format: "({}*sin({})+{})" -> (scaling, input, bias)
        """
        return [op.string_template for op in self.unary_operators]
    
    def get_unary_leaf_strings(self) -> List[str]:
        """
        Get string templates for unary operators at leaf nodes.
        
        These templates don't include scaling/bias:
        format: "(sin({}))"
        """
        return [op.get_leaf_template() for op in self.unary_operators]
    
    def get_binary_strings(self) -> List[str]:
        """
        Get string templates for binary operators.
        
        Templates format with (left, right) operands:
        format: "(({})+({}))"
        """
        return [op.string_template for op in self.binary_operators]
    
    def get_unary_names(self) -> List[str]:
        """Get list of unary operator names."""
        return [op.name for op in self.unary_operators]
    
    def get_binary_names(self) -> List[str]:
        """Get list of binary operator names."""
        return [op.name for op in self.binary_operators]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the grammar specification.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for empty operator sets
        if not self.unary_operators:
            errors.append("Grammar has no unary operators")
        if not self.binary_operators:
            errors.append("Grammar has no binary operators")
        
        # Check for duplicate names within each arity category
        unary_names = [op.name for op in self.unary_operators]
        binary_names = [op.name for op in self.binary_operators]
        
        seen_unary = set()
        for name in unary_names:
            if name in seen_unary:
                errors.append(f"Duplicate unary operator name: {name}")
            seen_unary.add(name)
        
        seen_binary = set()
        for name in binary_names:
            if name in seen_binary:
                errors.append(f"Duplicate binary operator name: {name}")
            seen_binary.add(name)
        
        # Validate each operator
        for op in self.unary_operators:
            if op.arity != "unary":
                errors.append(f"Operator '{op.name}' in unary list has arity '{op.arity}'")
        
        for op in self.binary_operators:
            if op.arity != "binary":
                errors.append(f"Operator '{op.name}' in binary list has arity '{op.arity}'")
        
        # Check implementations are callable
        for op in self.unary_operators + self.binary_operators:
            if not callable(op.implementation):
                errors.append(f"Operator '{op.name}' implementation is not callable")
        
        return len(errors) == 0, errors
    
    def get_operator_by_name(self, name: str) -> Optional[OperatorSpec]:
        """
        Find an operator by name (or alias).
        
        Args:
            name: Operator name or alias to search for
            
        Returns:
            OperatorSpec if found, None otherwise
        """
        name_lower = name.lower().strip()
        
        for op in self.unary_operators:
            if op.matches_token(name):
                return op
        
        for op in self.binary_operators:
            if op.matches_token(name):
                return op
        
        return None
    
    def get_operator_index(self, name: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Get the index and arity category of an operator.
        
        Args:
            name: Operator name or alias
            
        Returns:
            Tuple of (index, "unary"/"binary") or (None, None) if not found
        """
        name_lower = name.lower().strip()
        
        for idx, op in enumerate(self.unary_operators):
            if op.matches_token(name):
                return idx, "unary"
        
        for idx, op in enumerate(self.binary_operators):
            if op.matches_token(name):
                return idx, "binary"
        
        return None, None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert grammar to a dictionary representation (for serialization).
        
        Note: Implementation callables are not serialized; only names are preserved.
        """
        return {
            "name": self.name,
            "unary_operators": [
                {
                    "name": op.name,
                    "description": op.description,
                    "string_template": op.string_template,
                    "leaf_string_template": op.leaf_string_template,
                    "token_aliases": op.token_aliases,
                }
                for op in self.unary_operators
            ],
            "binary_operators": [
                {
                    "name": op.name,
                    "description": op.description,
                    "string_template": op.string_template,
                    "token_aliases": op.token_aliases,
                }
                for op in self.binary_operators
            ],
            "metadata": self.metadata,
        }
    
    @property
    def num_unary(self) -> int:
        """Number of unary operators."""
        return len(self.unary_operators)
    
    @property
    def num_binary(self) -> int:
        """Number of binary operators."""
        return len(self.binary_operators)
    
    def __repr__(self) -> str:
        return (
            f"GrammarSpec(name='{self.name}', "
            f"unary={self.num_unary}, binary={self.num_binary})"
        )
    
    def __str__(self) -> str:
        unary_names = ", ".join(self.get_unary_names())
        binary_names = ", ".join(self.get_binary_names())
        return (
            f"Grammar: {self.name}\n"
            f"  Unary operators ({self.num_unary}): {unary_names}\n"
            f"  Binary operators ({self.num_binary}): {binary_names}"
        )
