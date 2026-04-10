"""
Configuration handling for FEX grammar system.

This module provides configuration classes and utilities for
managing grammar settings via CLI arguments or configuration files.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .grammar_provider import (
    GrammarProvider,
    FEXGrammarProvider,
    LLMGrammarProvider,
    HybridGrammarProvider,
    create_grammar_provider,
)


@dataclass
class GrammarConfig:
    """
    Configuration for FEX grammar system.
    
    This class encapsulates all grammar-related configuration options
    and provides methods to create grammar providers from configuration.
    
    Attributes:
        source: Grammar source ("fex", "llm", or "hybrid")
        path: Path to LLM grammar file
        format: Format of LLM grammar file ("json", "yaml", "auto")
        strict_arity_validation: Fail on arity mismatches
        allow_unknown_tokens: Skip unknown tokens instead of failing
        fallback_to_fex: Fall back to FEX grammar on errors
        use_extended_fex: Use extended FEX vocabulary
    """
    source: str = "fex"
    path: Optional[str] = None
    format: str = "auto"
    strict_arity_validation: bool = True
    allow_unknown_tokens: bool = False
    fallback_to_fex: bool = False
    use_extended_fex: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.source = self.source.lower()
        
        valid_sources = ("fex", "llm", "hybrid")
        if self.source not in valid_sources:
            raise ValueError(
                f"Invalid grammar source '{self.source}'. "
                f"Must be one of: {valid_sources}"
            )
        
        if self.source in ("llm", "hybrid") and not self.path:
            raise ValueError(
                f"grammar_path is required for source '{self.source}'"
            )
    
    def create_provider(self) -> GrammarProvider:
        """
        Create a grammar provider from this configuration.
        
        Returns:
            GrammarProvider instance based on configuration
        """
        return create_grammar_provider(
            source=self.source,
            grammar_path=self.path,
            grammar_format=self.format,
            strict_arity_validation=self.strict_arity_validation,
            allow_unknown_tokens=self.allow_unknown_tokens,
            fallback_to_fex=self.fallback_to_fex,
            use_extended_fex=self.use_extended_fex,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "source": self.source,
            "path": self.path,
            "format": self.format,
            "strict_arity_validation": self.strict_arity_validation,
            "allow_unknown_tokens": self.allow_unknown_tokens,
            "fallback_to_fex": self.fallback_to_fex,
            "use_extended_fex": self.use_extended_fex,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrammarConfig":
        """Create configuration from dictionary."""
        return cls(
            source=data.get("source", "fex"),
            path=data.get("path"),
            format=data.get("format", "auto"),
            strict_arity_validation=data.get("strict_arity_validation", True),
            allow_unknown_tokens=data.get("allow_unknown_tokens", False),
            fallback_to_fex=data.get("fallback_to_fex", False),
            use_extended_fex=data.get("use_extended_fex", False),
        )
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GrammarConfig":
        """
        Create configuration from argparse Namespace.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            GrammarConfig instance
        """
        return cls(
            source=getattr(args, "grammar_source", "fex"),
            path=getattr(args, "grammar_path", None),
            format=getattr(args, "grammar_format", "auto"),
            strict_arity_validation=getattr(args, "strict_arity_validation", True),
            allow_unknown_tokens=getattr(args, "allow_unknown_tokens", False),
            fallback_to_fex=getattr(args, "grammar_fallback_to_fex", False),
            use_extended_fex=getattr(args, "use_extended_fex", False),
        )


def add_grammar_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add grammar-related arguments to an argument parser.
    
    This function adds the following arguments to the parser:
        --grammar_source: Grammar source (fex, llm, hybrid)
        --grammar_path: Path to LLM grammar file
        --grammar_format: Format of LLM grammar file
        --allow_unknown_tokens: Skip unknown tokens
        --strict_arity_validation: Enable/disable strict arity validation
        --grammar_fallback_to_fex: Fall back to FEX on errors
        --use_extended_fex: Use extended FEX vocabulary
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--grammar_source",
        type=str,
        default="fex",
        choices=["fex", "llm", "hybrid"],
        help=(
            "Grammar source for FEX operators. "
            "'fex' (default) uses the original FEX grammar. "
            "'llm' uses a grammar from an LLM-generated file. "
            "'hybrid' combines FEX and LLM grammars."
        ),
    )
    
    parser.add_argument(
        "--grammar_path",
        type=str,
        default="",
        help=(
            "Path to LLM grammar file (JSON or YAML). "
            "Required when grammar_source is 'llm' or 'hybrid'."
        ),
    )
    
    parser.add_argument(
        "--grammar_format",
        type=str,
        default="auto",
        choices=["json", "yaml", "auto"],
        help=(
            "Format of LLM grammar file. "
            "'auto' (default) detects format from file extension."
        ),
    )
    
    parser.add_argument(
        "--allow_unknown_tokens",
        action="store_true",
        help=(
            "Skip unknown operator tokens in LLM grammar instead of failing. "
            "Useful for grammars with operators not in FEX vocabulary."
        ),
    )
    
    parser.add_argument(
        "--strict_arity_validation",
        action="store_true",
        default=True,
        help=(
            "Enable strict arity validation (default: True). "
            "Reject operators in wrong arity category."
        ),
    )
    
    parser.add_argument(
        "--no_strict_arity_validation",
        action="store_false",
        dest="strict_arity_validation",
        help="Disable strict arity validation.",
    )
    
    parser.add_argument(
        "--grammar_fallback_to_fex",
        action="store_true",
        help=(
            "Fall back to FEX grammar if LLM grammar loading fails. "
            "If not set, LLM grammar errors will raise exceptions."
        ),
    )
    
    parser.add_argument(
        "--use_extended_fex",
        action="store_true",
        help=(
            "Use extended FEX vocabulary (includes additional operators "
            "like sqrt, tanh, sigmoid, log, reciprocal, etc.)"
        ),
    )


def load_grammar_from_config(
    config: GrammarConfig,
    verbose: bool = True,
) -> GrammarProvider:
    """
    Load a grammar provider from configuration.
    
    This is a convenience function that creates a grammar provider
    and optionally prints information about the loaded grammar.
    
    Args:
        config: GrammarConfig instance
        verbose: If True, print grammar information
        
    Returns:
        GrammarProvider instance
    """
    provider = config.create_provider()
    
    if verbose:
        grammar = provider.get_grammar()
        print(f"Loaded grammar: {grammar.name}")
        print(f"  Unary operators ({grammar.num_unary}): {grammar.get_unary_names()}")
        print(f"  Binary operators ({grammar.num_binary}): {grammar.get_binary_names()}")
    
    return provider


def load_grammar_from_args(
    args: argparse.Namespace,
    verbose: bool = True,
) -> GrammarProvider:
    """
    Load a grammar provider from command-line arguments.
    
    This is the main entry point for loading grammars in FEX scripts.
    
    Args:
        args: Parsed command-line arguments
        verbose: If True, print grammar information
        
    Returns:
        GrammarProvider instance
    """
    config = GrammarConfig.from_args(args)
    return load_grammar_from_config(config, verbose=verbose)
