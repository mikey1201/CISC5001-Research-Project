"""
Grammar abstraction module for FEX.

This module provides a flexible grammar system that allows FEX to use either:
1. The original fixed FEX grammar (default)
2. LLM-generated grammars
3. Hybrid grammars combining both

Key classes:
- GrammarSpec: Complete specification of operators for expression search
- GrammarProvider: Abstract base for grammar sources
- FEXGrammarProvider: Default FEX grammar provider
- LLMGrammarProvider: Provider for LLM-generated grammars
- LLMGrammarAdapter: Converts LLM outputs to GrammarSpec
"""

from .grammar_spec import GrammarSpec, OperatorSpec
from .grammar_provider import (
    GrammarProvider,
    FEXGrammarProvider,
    LLMGrammarProvider,
    HybridGrammarProvider,
    create_grammar_provider,
)
from .llm_grammar_adapter import LLMGrammarAdapter, LLMGrammarError
from .default_grammar import get_default_fex_grammar
from .config import GrammarConfig, load_grammar_from_config, load_grammar_from_args, add_grammar_arguments

__all__ = [
    # Core specification
    "GrammarSpec",
    "OperatorSpec",
    # Providers
    "GrammarProvider",
    "FEXGrammarProvider",
    "LLMGrammarProvider",
    "HybridGrammarProvider",
    "create_grammar_provider",
    # Adapter
    "LLMGrammarAdapter",
    "LLMGrammarError",
    # Default grammar
    "get_default_fex_grammar",
    # Configuration
    "GrammarConfig",
    "load_grammar_from_config",
    "load_grammar_from_args",
    "add_grammar_arguments",
]
