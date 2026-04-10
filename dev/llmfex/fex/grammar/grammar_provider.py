"""
Grammar Provider abstraction for FEX.

This module provides the abstraction layer for grammar sources,
allowing FEX to use different grammar sources (default FEX, LLM, hybrid)
through a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .grammar_spec import GrammarSpec
from .default_grammar import get_default_fex_grammar, get_extended_grammar
from .llm_grammar_adapter import LLMGrammarAdapter, LLMGrammarError


class GrammarProvider(ABC):
    """
    Abstract base class for grammar sources.
    
    A GrammarProvider is responsible for providing a GrammarSpec
    that defines the operator search space for FEX.
    """
    
    @abstractmethod
    def get_grammar(self) -> GrammarSpec:
        """
        Get the grammar specification.
        
        Returns:
            GrammarSpec object defining operators for expression search
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this grammar provider.
        
        Returns:
            Human-readable name string
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this grammar provider.
        
        Returns:
            Dictionary with provider metadata
        """
        grammar = self.get_grammar()
        return {
            "provider_name": self.get_name(),
            "grammar_name": grammar.name,
            "num_unary": grammar.num_unary,
            "num_binary": grammar.num_binary,
            "metadata": grammar.metadata,
        }


class FEXGrammarProvider(GrammarProvider):
    """
    Provider for the default FEX grammar.
    
    This is the original FEX operator library as defined in the
    Finite Expression Method paper. This provider is used when
    no LLM grammar is specified.
    
    Attributes:
        use_extended: If True, use the extended operator vocabulary
    """
    
    def __init__(self, use_extended: bool = False):
        """
        Initialize the FEX grammar provider.
        
        Args:
            use_extended: If True, use extended vocabulary with additional operators
        """
        self.use_extended = use_extended
        self._grammar: Optional[GrammarSpec] = None
    
    def get_grammar(self) -> GrammarSpec:
        """Get the FEX grammar specification."""
        if self._grammar is None:
            if self.use_extended:
                self._grammar = get_extended_grammar()
            else:
                self._grammar = get_default_fex_grammar()
        return self._grammar
    
    def get_name(self) -> str:
        """Get the provider name."""
        return "FEX_Default" if not self.use_extended else "FEX_Extended"


class LLMGrammarProvider(GrammarProvider):
    """
    Provider for LLM-generated grammars.
    
    This provider loads grammars from external files that are generated
    by fine-tuned LLMs. The grammar specifies which operators should be
    used for a particular PDE type.
    
    Attributes:
        grammar_path: Path to the LLM grammar file
        grammar_format: Format of the grammar file ("json", "yaml", "auto")
        strict_arity_validation: Fail on arity mismatches
        allow_unknown_tokens: Skip unknown tokens instead of failing
        fallback_to_fex: Fall back to FEX grammar on errors
    """
    
    def __init__(
        self,
        grammar_path: str,
        grammar_format: str = "auto",
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
        fallback_to_fex: bool = False,
    ):
        """
        Initialize the LLM grammar provider.
        
        Args:
            grammar_path: Path to LLM grammar file
            grammar_format: File format ("json", "yaml", or "auto")
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            fallback_to_fex: If True, fall back to FEX grammar on load errors
        """
        self.grammar_path = grammar_path
        self.grammar_format = grammar_format
        self.strict_arity_validation = strict_arity_validation
        self.allow_unknown_tokens = allow_unknown_tokens
        self.fallback_to_fex = fallback_to_fex
        self._grammar: Optional[GrammarSpec] = None
        self._load_error: Optional[str] = None
    
    def get_grammar(self) -> GrammarSpec:
        """
        Get the LLM grammar specification.
        
        Returns:
            GrammarSpec from the LLM grammar file
            
        Raises:
            LLMGrammarError: If grammar cannot be loaded and fallback is disabled
        """
        if self._grammar is not None:
            return self._grammar
        
        try:
            self._grammar = LLMGrammarAdapter.from_file(
                path=self.grammar_path,
                format=self.grammar_format,
                strict_arity_validation=self.strict_arity_validation,
                allow_unknown_tokens=self.allow_unknown_tokens,
            )
            return self._grammar
        except LLMGrammarError as e:
            self._load_error = str(e)
            if self.fallback_to_fex:
                print(f"Warning: Failed to load LLM grammar, falling back to FEX: {e}")
                self._grammar = get_default_fex_grammar()
                return self._grammar
            raise
    
    def get_name(self) -> str:
        """Get the provider name."""
        return f"LLM_{self.grammar_path}"
    
    def get_load_error(self) -> Optional[str]:
        """
        Get the error message if grammar loading failed.
        
        Returns:
            Error message string, or None if no error
        """
        return self._load_error
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = False,
    ) -> "LLMGrammarProvider":
        """
        Create an LLM grammar provider from a dictionary.
        
        This creates an in-memory provider without a file.
        
        Args:
            data: Dictionary containing grammar specification
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            
        Returns:
            LLMGrammarProvider instance
        """
        # Create a provider with dummy path, then load from dict
        provider = cls(
            grammar_path=":memory:",
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
        )
        provider._grammar = LLMGrammarAdapter.from_dict(
            data,
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
        )
        return provider


class HybridGrammarProvider(GrammarProvider):
    """
    Provider that combines FEX and LLM grammars.
    
    This provider merges the default FEX grammar with an LLM-generated
    grammar, ensuring that the FEX operators are always available while
    adding any additional operators from the LLM.
    
    This is useful when:
    - The LLM grammar might be incomplete
    - You want to ensure basic operators are always available
    - You're experimenting with LLM grammars and want a safety net
    
    Attributes:
        fex_provider: The base FEX grammar provider
        llm_provider: The LLM grammar provider (optional)
    """
    
    def __init__(
        self,
        llm_grammar_path: Optional[str] = None,
        llm_grammar_format: str = "auto",
        llm_grammar_dict: Optional[Dict[str, Any]] = None,
        strict_arity_validation: bool = True,
        allow_unknown_tokens: bool = True,
        use_extended_fex: bool = False,
    ):
        """
        Initialize the hybrid grammar provider.
        
        Args:
            llm_grammar_path: Path to LLM grammar file (optional)
            llm_grammar_format: Format of LLM grammar file
            llm_grammar_dict: LLM grammar as dictionary (alternative to file)
            strict_arity_validation: Fail on arity mismatches
            allow_unknown_tokens: Skip unknown tokens instead of failing
            use_extended_fex: Use extended FEX vocabulary as base
        """
        self.fex_provider = FEXGrammarProvider(use_extended=use_extended_fex)
        
        self.llm_provider: Optional[LLMGrammarProvider] = None
        if llm_grammar_path:
            self.llm_provider = LLMGrammarProvider(
                grammar_path=llm_grammar_path,
                grammar_format=llm_grammar_format,
                strict_arity_validation=strict_arity_validation,
                allow_unknown_tokens=allow_unknown_tokens,
                fallback_to_fex=False,  # Don't double-fallback
            )
        elif llm_grammar_dict:
            self.llm_provider = LLMGrammarProvider.from_dict(
                llm_grammar_dict,
                strict_arity_validation=strict_arity_validation,
                allow_unknown_tokens=allow_unknown_tokens,
            )
        
        self._grammar: Optional[GrammarSpec] = None
    
    def get_grammar(self) -> GrammarSpec:
        """
        Get the hybrid grammar specification.
        
        Returns:
            GrammarSpec combining FEX and LLM operators
        """
        if self._grammar is not None:
            return self._grammar
        
        # Start with FEX grammar
        fex_grammar = self.fex_provider.get_grammar()
        
        # If no LLM grammar, just return FEX
        if self.llm_provider is None:
            self._grammar = fex_grammar
            return self._grammar
        
        # Try to load LLM grammar
        try:
            llm_grammar = self.llm_provider.get_grammar()
        except LLMGrammarError as e:
            print(f"Warning: Failed to load LLM grammar for hybrid, using FEX only: {e}")
            self._grammar = fex_grammar
            return self._grammar
        
        # Merge operators (avoid duplicates by name)
        fex_unary_names = {op.name for op in fex_grammar.unary_operators}
        fex_binary_names = {op.name for op in fex_grammar.binary_operators}
        
        unary_operators = list(fex_grammar.unary_operators)
        for op in llm_grammar.unary_operators:
            if op.name not in fex_unary_names:
                unary_operators.append(op)
        
        binary_operators = list(fex_grammar.binary_operators)
        for op in llm_grammar.binary_operators:
            if op.name not in fex_binary_names:
                binary_operators.append(op)
        
        # Create merged grammar
        self._grammar = GrammarSpec(
            name=f"Hybrid_{fex_grammar.name}_{llm_grammar.name}",
            unary_operators=unary_operators,
            binary_operators=binary_operators,
            metadata={
                "source": "hybrid",
                "fex_source": fex_grammar.metadata,
                "llm_source": llm_grammar.metadata,
            },
        )
        
        return self._grammar
    
    def get_name(self) -> str:
        """Get the provider name."""
        return "Hybrid_FEX_LLM"


def create_grammar_provider(
    source: str = "fex",
    grammar_path: Optional[str] = None,
    grammar_format: str = "auto",
    strict_arity_validation: bool = True,
    allow_unknown_tokens: bool = False,
    fallback_to_fex: bool = False,
    use_extended_fex: bool = False,
) -> GrammarProvider:
    """
    Factory function to create a grammar provider.
    
    This is the recommended way to create grammar providers based on
    configuration parameters.
    
    Args:
        source: Grammar source ("fex", "llm", or "hybrid")
        grammar_path: Path to LLM grammar file (required for "llm" source)
        grammar_format: Format of LLM grammar file
        strict_arity_validation: Fail on arity mismatches
        allow_unknown_tokens: Skip unknown tokens instead of failing
        fallback_to_fex: Fall back to FEX grammar on errors
        use_extended_fex: Use extended FEX vocabulary
        
    Returns:
        GrammarProvider instance
        
    Raises:
        ValueError: If invalid source or missing required parameters
    """
    source = source.lower()
    
    if source == "fex":
        return FEXGrammarProvider(use_extended=use_extended_fex)
    
    elif source == "llm":
        if not grammar_path:
            raise ValueError("grammar_path is required for 'llm' grammar source")
        return LLMGrammarProvider(
            grammar_path=grammar_path,
            grammar_format=grammar_format,
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
            fallback_to_fex=fallback_to_fex,
        )
    
    elif source == "hybrid":
        return HybridGrammarProvider(
            llm_grammar_path=grammar_path,
            llm_grammar_format=grammar_format,
            strict_arity_validation=strict_arity_validation,
            allow_unknown_tokens=allow_unknown_tokens,
            use_extended_fex=use_extended_fex,
        )
    
    else:
        raise ValueError(
            f"Invalid grammar source: '{source}'. "
            "Must be one of: 'fex', 'llm', 'hybrid'"
        )
