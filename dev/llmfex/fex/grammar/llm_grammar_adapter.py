"""
LLM Grammar Adapter for FEX.

This module provides utilities for converting LLM-generated grammars
(in various formats) into FEX-compatible GrammarSpec objects.

The adapter handles:
- JSON/YAML grammar file loading
- Postfix token sequence parsing
- Duplicate token removal
- Invalid token rejection
- Operator vocabulary mapping
- Arity validation
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from .grammar_spec import GrammarSpec, OperatorSpec
from .default_grammar import (
    get_operator_vocabulary,
    DEFAULT_UNARY_OPERATORS,
    DEFAULT_BINARY_OPERATORS,
)


class LLMGrammarError(Exception):
    """Exception raised for LLM grammar parsing errors."""
    pass


class LLMGrammarAdapter:
    """
    Adapter for converting LLM outputs to FEX GrammarSpec.
    
    This class provides methods to parse various LLM output formats
    and convert them into valid FEX grammars.
    
    Attributes:
        OPERATOR_VOCABULARY: Mapping of token names to OperatorSpec objects
        strict_arity_validation: If True, reject tokens with wrong arity
        allow_unknown_tokens: If True, skip unknown tokens instead of failing
    """
    
    # Pre-computed vocabulary for fast lookup
    OPERATOR_VOCABULARY: Dict[str, OperatorSpec] = get_operator_vocabulary()
    
    def __init__(
        self,
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
    ):
        """
        Initialize the LLM grammar adapter.
        
        Args:
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
        """
        self.strict_arity_validation = strict_arity_validation
        self.allow_unknown_tokens = allow_unknown_tokens
    
    @classmethod
    def from_json_file(
        cls,
        path: str,
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
    ) -> GrammarSpec:
        """
        Load a grammar from a JSON file.
        
        Expected JSON format:
        {
            "grammar_type": "llm_generated",
            "unary_operators": [
                {"name": "sin"},
                {"name": "exp"}
            ],
            "binary_operators": [
                {"name": "add"},
                {"name": "mul"}
            ],
            "metadata": {...}  // optional
        }
        
        Args:
            path: Path to JSON file
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            
        Returns:
            GrammarSpec object
            
        Raises:
            LLMGrammarError: If file cannot be parsed or is invalid
        """
        if not os.path.exists(path):
            raise LLMGrammarError(f"Grammar file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise LLMGrammarError(f"Invalid JSON in grammar file: {e}")
        
        return cls.from_dict(
            data,
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
        )
    
    @classmethod
    def from_yaml_file(
        cls,
        path: str,
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
    ) -> GrammarSpec:
        """
        Load a grammar from a YAML file.
        
        Args:
            path: Path to YAML file
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            
        Returns:
            GrammarSpec object
            
        Raises:
            LLMGrammarError: If file cannot be parsed or is invalid
        """
        try:
            import yaml
        except ImportError:
            raise LLMGrammarError(
                "PyYAML is required for YAML grammar files. "
                "Install with: pip install pyyaml"
            )
        
        if not os.path.exists(path):
            raise LLMGrammarError(f"Grammar file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise LLMGrammarError(f"Invalid YAML in grammar file: {e}")
        
        return cls.from_dict(
            data,
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
        )
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
    ) -> GrammarSpec:
        """
        Create a grammar from a dictionary.
        
        Expected dictionary format:
        {
            "unary_operators": [{"name": "sin"}, ...],
            "binary_operators": [{"name": "add"}, ...],
            "metadata": {...}  // optional
        }
        
        Or for postfix token format:
        {
            "format": "postfix_tokens",
            "tokens": ["sin", "add", "exp", ...],
            "metadata": {...}  // optional
        }
        
        Args:
            data: Dictionary containing grammar specification
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            
        Returns:
            GrammarSpec object
        """
        adapter = cls(
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
        )
        
        # Check for postfix token format
        if data.get("format") == "postfix_tokens":
            return adapter.from_postfix_tokens(data.get("tokens", []), data.get("metadata", {}))
        
        # Standard operator list format
        unary_tokens = []
        for op in data.get("unary_operators", []):
            if isinstance(op, str):
                unary_tokens.append(op)
            elif isinstance(op, dict):
                unary_tokens.append(op.get("name", ""))
        
        binary_tokens = []
        for op in data.get("binary_operators", []):
            if isinstance(op, str):
                binary_tokens.append(op)
            elif isinstance(op, dict):
                binary_tokens.append(op.get("name", ""))
        
        return adapter.build_grammar(
            unary_tokens=unary_tokens,
            binary_tokens=binary_tokens,
            metadata=data.get("metadata", {}),
            name=data.get("name", "LLM_Grammar"),
        )
    
    @classmethod
    def from_postfix_tokens(
        cls,
        tokens: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
    ) -> GrammarSpec:
        """
        Create a grammar from a postfix token sequence.
        
        This extracts unique operators from a postfix (RPN) expression
        and builds a grammar containing only those operators.
        
        Args:
            tokens: List of tokens in postfix notation
            metadata: Optional metadata dictionary
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            
        Returns:
            GrammarSpec containing extracted operators
        """
        adapter = cls(
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
        )
        
        # Separate tokens by arity
        unary_tokens = set()
        binary_tokens = set()
        
        valid, unknown, arity_errors = adapter.validate_tokens(tokens)
        
        for token in valid:
            op_spec = adapter.OPERATOR_VOCABULARY.get(token.lower())
            if op_spec:
                if op_spec.arity == "unary":
                    unary_tokens.add(token.lower())
                else:
                    binary_tokens.add(token.lower())
        
        return adapter.build_grammar(
            unary_tokens=list(unary_tokens),
            binary_tokens=list(binary_tokens),
            metadata=metadata or {},
            name="LLM_Postfix_Grammar",
        )
    
    def validate_tokens(
        self,
        tokens: List[str],
    ) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        """
        Validate a list of tokens against the FEX vocabulary.
        
        Args:
            tokens: List of token strings to validate
            
        Returns:
            Tuple of (valid_tokens, unknown_tokens, arity_errors)
            where arity_errors is a list of (token, expected_arity, actual_arity)
        """
        valid = []
        unknown = []
        arity_errors = []
        
        for token in tokens:
            token_lower = token.lower().strip()
            op_spec = self.OPERATOR_VOCABULARY.get(token_lower)
            
            if op_spec is None:
                if self.allow_unknown_tokens:
                    unknown.append(token)
                else:
                    raise LLMGrammarError(
                        f"Unknown token '{token}'. "
                        f"Known operators: {list(self.OPERATOR_VOCABULARY.keys())}"
                    )
            else:
                valid.append(token_lower)
        
        return valid, unknown, arity_errors
    
    def remove_duplicates(self, tokens: List[str]) -> List[str]:
        """
        Remove duplicate tokens while preserving order.
        
        Args:
            tokens: List of tokens (may contain duplicates)
            
        Returns:
            List with duplicates removed, order preserved
        """
        seen = set()
        result = []
        for token in tokens:
            token_lower = token.lower().strip()
            if token_lower not in seen:
                seen.add(token_lower)
                result.append(token_lower)
        return result
    
    def map_to_vocabulary(
        self,
        tokens: List[str],
        expected_arity: Optional[str] = None,
    ) -> List[OperatorSpec]:
        """
        Map token names to OperatorSpec objects.
        
        Args:
            tokens: List of token names
            expected_arity: If provided, validate that tokens match this arity
            
        Returns:
            List of OperatorSpec objects
            
        Raises:
            LLMGrammarError: If token not found or arity mismatch
        """
        operators = []
        
        for token in tokens:
            token_lower = token.lower().strip()
            op_spec = self.OPERATOR_VOCABULARY.get(token_lower)
            
            if op_spec is None:
                if not self.allow_unknown_tokens:
                    raise LLMGrammarError(f"Unknown operator token: '{token}'")
                continue
            
            # Validate arity if requested
            if expected_arity and op_spec.arity != expected_arity:
                if self.strict_arity_validation:
                    raise LLMGrammarError(
                        f"Arity mismatch for '{token}': "
                        f"expected {expected_arity}, got {op_spec.arity}"
                    )
                else:
                    continue  # Skip this operator
            
            operators.append(op_spec)
        
        return operators
    
    def build_grammar(
        self,
        unary_tokens: List[str],
        binary_tokens: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "LLM_Grammar",
    ) -> GrammarSpec:
        """
        Build a GrammarSpec from token lists.
        
        Args:
            unary_tokens: List of unary operator token names
            binary_tokens: List of binary operator token names
            metadata: Optional metadata dictionary
            name: Name for the grammar
            
        Returns:
            GrammarSpec object
            
        Raises:
            LLMGrammarError: If grammar is empty or invalid
        """
        # Remove duplicates
        unary_tokens = self.remove_duplicates(unary_tokens)
        binary_tokens = self.remove_duplicates(binary_tokens)
        
        # Map to operator specs
        unary_operators = self.map_to_vocabulary(unary_tokens, expected_arity="unary")
        binary_operators = self.map_to_vocabulary(binary_tokens, expected_arity="binary")
        
        # Validate
        if not unary_operators:
            raise LLMGrammarError(
                "Grammar has no valid unary operators. "
                f"Provided tokens: {unary_tokens}"
            )
        if not binary_operators:
            raise LLMGrammarError(
                "Grammar has no valid binary operators. "
                f"Provided tokens: {binary_tokens}"
            )
        
        return GrammarSpec(
            name=name,
            unary_operators=unary_operators,
            binary_operators=binary_operators,
            metadata=metadata or {
                "source": "llm_generated",
                "version": "1.0",
            },
        )
    
    @classmethod
    def from_file(
        cls,
        path: str,
        format: str = "json",
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
    ) -> GrammarSpec:
        """
        Load a grammar from a file (auto-detect format if not specified).
        
        Args:
            path: Path to grammar file
            format: File format ("json", "yaml", or "auto")
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            
        Returns:
            GrammarSpec object
        """
        # Auto-detect format from extension
        if format == "auto":
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext == ".json":
                format = "json"
            elif ext in (".yaml", ".yml"):
                format = "yaml"
            else:
                raise LLMGrammarError(
                    f"Cannot auto-detect format for file: {path}. "
                    "Please specify format explicitly."
                )
        
        if format == "json":
            return cls.from_json_file(
                path,
                strict_arity_validation=strict_arity_validation,
                allow_unknown_tokens=allow_unknown_tokens,
            )
        elif format == "yaml":
            return cls.from_yaml_file(
                path,
                strict_arity_validation=strict_arity_validation,
                allow_unknown_tokens=allow_unknown_tokens,
            )
        else:
            raise LLMGrammarError(f"Unsupported grammar format: {format}")
    
    @classmethod
    def get_supported_operators(cls) -> Dict[str, List[str]]:
        """
        Get a dictionary of supported operators grouped by arity.
        
        Returns:
            Dictionary with 'unary' and 'binary' keys, each mapping to
            a list of operator names
        """
        unary = set()
        binary = set()
        
        for name, op_spec in cls.OPERATOR_VOCABULARY.items():
            if op_spec.arity == "unary":
                unary.add(op_spec.name)
            else:
                binary.add(op_spec.name)
        
        return {
            "unary": sorted(list(unary)),
            "binary": sorted(list(binary)),
        }
    
    @classmethod
    def print_supported_operators(cls) -> None:
        """Print the list of supported operators."""
        operators = cls.get_supported_operators()
        print("Supported FEX Operators:")
        print("\nUnary operators:")
        for op in operators["unary"]:
            print(f"  - {op}")
        print("\nBinary operators:")
        for op in operators["binary"]:
            print(f"  - {op}")
