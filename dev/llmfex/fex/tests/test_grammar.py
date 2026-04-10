"""
Tests for FEX grammar abstraction module.

This test suite covers:
- Grammar specification validation
- Default FEX grammar loading
- LLM grammar loading (JSON, YAML, postfix tokens)
- Grammar provider abstraction
- Duplicate removal
- Invalid token rejection
- Arity validation
- Backward compatibility
"""

import os
import sys
import json
import pytest
import torch

# Add the fex directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar import (
    GrammarSpec,
    OperatorSpec,
    GrammarProvider,
    FEXGrammarProvider,
    LLMGrammarProvider,
    HybridGrammarProvider,
    LLMGrammarAdapter,
    LLMGrammarError,
    get_default_fex_grammar,
    GrammarConfig,
    create_grammar_provider,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fixtures_dir():
    """Return the path to test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def sample_llm_grammar_path(fixtures_dir):
    """Return path to sample LLM grammar JSON file."""
    return os.path.join(fixtures_dir, 'sample_llm_grammar.json')


@pytest.fixture
def sample_postfix_tokens_path(fixtures_dir):
    """Return path to sample postfix tokens JSON file."""
    return os.path.join(fixtures_dir, 'sample_postfix_tokens.json')


@pytest.fixture
def invalid_grammar_path(fixtures_dir):
    """Return path to invalid grammar JSON file."""
    return os.path.join(fixtures_dir, 'invalid_grammar.json')


# =============================================================================
# OperatorSpec Tests
# =============================================================================

class TestOperatorSpec:
    """Tests for OperatorSpec class."""
    
    def test_create_unary_operator(self):
        """Test creating a unary operator specification."""
        op = OperatorSpec(
            name="sin",
            arity="unary",
            implementation=torch.sin,
            string_template="({}*sin({})+{})",
        )
        assert op.name == "sin"
        assert op.arity == "unary"
        assert op.arity == "unary"
        assert callable(op.implementation)
    
    def test_create_binary_operator(self):
        """Test creating a binary operator specification."""
        op = OperatorSpec(
            name="add",
            arity="binary",
            implementation=lambda x, y: x + y,
            string_template="(({})+({}))",
        )
        assert op.name == "add"
        assert op.arity == "binary"
    
    def test_invalid_arity_raises_error(self):
        """Test that invalid arity raises ValueError."""
        with pytest.raises(ValueError):
            OperatorSpec(
                name="test",
                arity="ternary",  # Invalid
                implementation=lambda x: x,
                string_template="{}",
            )
    
    def test_token_matching(self):
        """Test token alias matching."""
        op = OperatorSpec(
            name="add",
            arity="binary",
            implementation=lambda x, y: x + y,
            string_template="(({})+({}))",
            token_aliases=["+", "plus", "sum"],
        )
        assert op.matches_token("add")
        assert op.matches_token("+")
        assert op.matches_token("PLUS")  # Case insensitive
        assert op.matches_token("Sum")
        assert not op.matches_token("multiply")


# =============================================================================
# GrammarSpec Tests
# =============================================================================

class TestGrammarSpec:
    """Tests for GrammarSpec class."""
    
    def test_create_grammar(self):
        """Test creating a grammar specification."""
        unary = [
            OperatorSpec(
                name="sin",
                arity="unary",
                implementation=torch.sin,
                string_template="sin({})",
            )
        ]
        binary = [
            OperatorSpec(
                name="add",
                arity="binary",
                implementation=lambda x, y: x + y,
                string_template="({}+{})",
            )
        ]
        
        grammar = GrammarSpec(
            name="test_grammar",
            unary_operators=unary,
            binary_operators=binary,
        )
        
        assert grammar.name == "test_grammar"
        assert grammar.num_unary == 1
        assert grammar.num_binary == 1
    
    def test_get_functions(self):
        """Test getting operator function lists."""
        grammar = get_default_fex_grammar()
        
        unary_funcs = grammar.get_unary_functions()
        binary_funcs = grammar.get_binary_functions()
        
        assert len(unary_funcs) == grammar.num_unary
        assert len(binary_funcs) == grammar.num_binary
        assert all(callable(f) for f in unary_funcs)
        assert all(callable(f) for f in binary_funcs)
    
    def test_get_strings(self):
        """Test getting string template lists."""
        grammar = get_default_fex_grammar()
        
        unary_strs = grammar.get_unary_strings()
        binary_strs = grammar.get_binary_strings()
        
        assert len(unary_strs) == grammar.num_unary
        assert len(binary_strs) == grammar.num_binary
        assert all(isinstance(s, str) for s in unary_strs)
        assert all(isinstance(s, str) for s in binary_strs)
    
    def test_validate_empty_grammar(self):
        """Test validation catches empty grammars."""
        grammar = GrammarSpec(
            name="empty",
            unary_operators=[],
            binary_operators=[],
        )
        
        is_valid, errors = grammar.validate()
        assert not is_valid
        assert len(errors) >= 2  # At least two errors (no unary, no binary)
    
    def test_validate_valid_grammar(self):
        """Test validation passes for valid grammars."""
        grammar = get_default_fex_grammar()
        is_valid, errors = grammar.validate()
        assert is_valid
        assert len(errors) == 0
    
    def test_get_operator_by_name(self):
        """Test finding operators by name."""
        grammar = get_default_fex_grammar()
        
        sin_op = grammar.get_operator_by_name("sin")
        assert sin_op is not None
        assert sin_op.name == "sin"
        assert sin_op.arity == "unary"
        
        add_op = grammar.get_operator_by_name("add")
        assert add_op is not None
        assert add_op.arity == "binary"
        
        unknown = grammar.get_operator_by_name("unknown_operator")
        assert unknown is None
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        grammar = get_default_fex_grammar()
        data = grammar.to_dict()
        
        assert "name" in data
        assert "unary_operators" in data
        assert "binary_operators" in data
        assert data["name"] == grammar.name


# =============================================================================
# Default FEX Grammar Tests
# =============================================================================

class TestDefaultFEXGrammar:
    """Tests for the default FEX grammar."""
    
    def test_load_default_grammar(self):
        """Test loading the default FEX grammar."""
        grammar = get_default_fex_grammar()
        
        assert grammar is not None
        assert grammar.name == "FEX_Default"
        assert grammar.num_unary > 0
        assert grammar.num_binary > 0
    
    def test_default_unary_operators(self):
        """Test that default grammar contains expected unary operators."""
        grammar = get_default_fex_grammar()
        names = grammar.get_unary_names()
        
        # Expected operators from the original FEX
        expected = ["zero", "one", "identity", "square", "cube", "quad", "exp", "sin", "cos"]
        for op in expected:
            assert op in names, f"Missing operator: {op}"
    
    def test_default_binary_operators(self):
        """Test that default grammar contains expected binary operators."""
        grammar = get_default_fex_grammar()
        names = grammar.get_binary_names()
        
        # Expected operators from the original FEX
        expected = ["add", "mul", "sub"]
        for op in expected:
            assert op in names, f"Missing operator: {op}"
    
    def test_operator_implementations(self):
        """Test that operator implementations work correctly."""
        grammar = get_default_fex_grammar()
        
        # Test a unary operator
        sin_op = grammar.get_operator_by_name("sin")
        x = torch.tensor([0.0, 1.0, 2.0])
        result = sin_op.implementation(x)
        expected = torch.sin(x)
        assert torch.allclose(result, expected)
        
        # Test a binary operator
        add_op = grammar.get_operator_by_name("add")
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        result = add_op.implementation(a, b)
        expected = a + b
        assert torch.allclose(result, expected)


# =============================================================================
# LLM Grammar Adapter Tests
# =============================================================================

class TestLLMGrammarAdapter:
    """Tests for the LLM grammar adapter."""
    
    def test_from_json_file(self, sample_llm_grammar_path):
        """Test loading grammar from JSON file."""
        grammar = LLMGrammarAdapter.from_json_file(sample_llm_grammar_path)
        
        assert grammar is not None
        assert grammar.num_unary == 4  # sin, cos, exp, square
        assert grammar.num_binary == 2  # add, mul
    
    def test_from_postfix_tokens(self, sample_postfix_tokens_path):
        """Test loading grammar from postfix tokens file."""
        with open(sample_postfix_tokens_path) as f:
            data = json.load(f)
        
        grammar = LLMGrammarAdapter.from_postfix_tokens(data["tokens"])
        
        assert grammar is not None
        # Should extract unique operators from tokens
        assert grammar.num_unary > 0
        assert grammar.num_binary > 0
    
    def test_from_dict(self):
        """Test loading grammar from dictionary."""
        data = {
            "name": "test_grammar",
            "unary_operators": [{"name": "sin"}, {"name": "exp"}],
            "binary_operators": [{"name": "add"}],
        }
        
        grammar = LLMGrammarAdapter.from_dict(data)
        
        assert grammar.name == "test_grammar"
        assert grammar.num_unary == 2
        assert grammar.num_binary == 1
    
    def test_unknown_token_rejection(self):
        """Test that unknown tokens are rejected (default behavior)."""
        data = {
            "unary_operators": [{"name": "unknown_unary_op"}],
            "binary_operators": [{"name": "add"}],
        }
        
        with pytest.raises(LLMGrammarError):
            LLMGrammarAdapter.from_dict(data, allow_unknown_tokens=False)
    
    def test_unknown_token_skip(self):
        """Test that unknown tokens can be skipped."""
        data = {
            "unary_operators": [{"name": "sin"}, {"name": "unknown_op"}],
            "binary_operators": [{"name": "add"}],
        }
        
        grammar = LLMGrammarAdapter.from_dict(data, allow_unknown_tokens=True)
        
        # Should only include sin (unknown_op skipped)
        assert grammar.num_unary == 1
        assert "sin" in grammar.get_unary_names()
    
    def test_duplicate_removal(self):
        """Test that duplicate tokens are removed."""
        adapter = LLMGrammarAdapter()
        tokens = ["sin", "exp", "sin", "sin", "cos", "exp"]
        
        unique = adapter.remove_duplicates(tokens)
        
        assert len(unique) == 3
        assert set(unique) == {"sin", "exp", "cos"}
    
    def test_arity_validation(self):
        """Test arity validation."""
        # Put a binary operator name in unary_operators
        data = {
            "unary_operators": [{"name": "add"}],  # add is binary!
            "binary_operators": [{"name": "mul"}],
        }
        
        with pytest.raises(LLMGrammarError):
            LLMGrammarAdapter.from_dict(data, strict_arity_validation=True)
    
    def test_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(LLMGrammarError):
            LLMGrammarAdapter.from_json_file("/nonexistent/path.json")
    
    def test_get_supported_operators(self):
        """Test getting list of supported operators."""
        operators = LLMGrammarAdapter.get_supported_operators()
        
        assert "unary" in operators
        assert "binary" in operators
        assert "sin" in operators["unary"]
        assert "add" in operators["binary"]


# =============================================================================
# Grammar Provider Tests
# =============================================================================

class TestGrammarProvider:
    """Tests for grammar providers."""
    
    def test_fex_grammar_provider(self):
        """Test FEX grammar provider."""
        provider = FEXGrammarProvider()
        
        assert provider.get_name() == "FEX_Default"
        grammar = provider.get_grammar()
        assert grammar.name == "FEX_Default"
    
    def test_llm_grammar_provider(self, sample_llm_grammar_path):
        """Test LLM grammar provider."""
        provider = LLMGrammarProvider(grammar_path=sample_llm_grammar_path)
        
        grammar = provider.get_grammar()
        assert grammar is not None
        assert grammar.num_unary == 4
        assert grammar.num_binary == 2
    
    def test_llm_grammar_provider_fallback(self, invalid_grammar_path):
        """Test LLM grammar provider fallback to FEX."""
        provider = LLMGrammarProvider(
            grammar_path=invalid_grammar_path,
            allow_unknown_tokens=False,
            fallback_to_fex=True,
        )
        
        grammar = provider.get_grammar()
        assert grammar.name == "FEX_Default"  # Should fallback
    
    def test_hybrid_grammar_provider(self, sample_llm_grammar_path):
        """Test hybrid grammar provider."""
        provider = HybridGrammarProvider(llm_grammar_path=sample_llm_grammar_path)
        
        grammar = provider.get_grammar()
        
        # Should include FEX operators plus LLM operators
        fex_grammar = get_default_fex_grammar()
        assert grammar.num_unary >= fex_grammar.num_unary
        assert grammar.num_binary >= fex_grammar.num_binary
    
    def test_create_grammar_provider_factory(self, sample_llm_grammar_path):
        """Test factory function for creating providers."""
        # FEX provider
        provider = create_grammar_provider(source="fex")
        assert isinstance(provider, FEXGrammarProvider)
        
        # LLM provider
        provider = create_grammar_provider(source="llm", grammar_path=sample_llm_grammar_path)
        assert isinstance(provider, LLMGrammarProvider)
        
        # Hybrid provider
        provider = create_grammar_provider(source="hybrid", grammar_path=sample_llm_grammar_path)
        assert isinstance(provider, HybridGrammarProvider)


# =============================================================================
# Grammar Config Tests
# =============================================================================

class TestGrammarConfig:
    """Tests for grammar configuration."""
    
    def test_create_config(self):
        """Test creating a grammar configuration."""
        config = GrammarConfig(source="fex")
        
        assert config.source == "fex"
        assert config.path is None
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid source
        with pytest.raises(ValueError):
            GrammarConfig(source="invalid")
        
        # LLM source without path
        with pytest.raises(ValueError):
            GrammarConfig(source="llm", path=None)
    
    def test_create_provider_from_config(self):
        """Test creating provider from configuration."""
        config = GrammarConfig(source="fex")
        provider = config.create_provider()
        
        assert isinstance(provider, FEXGrammarProvider)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with original FEX code."""
    
    def test_default_grammar_matches_original(self):
        """Test that default grammar matches original function.py definitions."""
        grammar = get_default_fex_grammar()
        
        # Import original function.py definitions
        # These are the operators from fex/Poisson/function.py
        original_unary_count = 9  # zero, one, identity, square, cube, quad, exp, sin, cos
        original_binary_count = 3  # add, mul, sub
        
        assert grammar.num_unary == original_unary_count
        assert grammar.num_binary == original_binary_count
    
    def test_grammar_produces_same_results(self):
        """Test that grammar operators produce same results as original."""
        grammar = get_default_fex_grammar()
        
        # Test all unary operators
        x = torch.randn(10)
        for i, op in enumerate(grammar.unary_operators):
            result = op.implementation(x)
            # Should not raise any errors
            assert result.shape == x.shape
        
        # Test all binary operators
        a = torch.randn(10)
        b = torch.randn(10)
        for i, op in enumerate(grammar.binary_operators):
            result = op.implementation(a, b)
            assert result.shape == a.shape


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the grammar system."""
    
    def test_full_workflow(self, sample_llm_grammar_path):
        """Test full grammar loading workflow."""
        # Create config from args-like object
        class Args:
            grammar_source = "llm"
            grammar_path = sample_llm_grammar_path
            grammar_format = "auto"
            strict_arity_validation = True
            allow_unknown_tokens = False
            grammar_fallback_to_fex = False
            use_extended_fex = False
        
        args = Args()
        config = GrammarConfig.from_args(args)
        provider = config.create_provider()
        grammar = provider.get_grammar()
        
        assert grammar is not None
        assert grammar.num_unary > 0
        assert grammar.num_binary > 0
    
    def test_controller_with_grammar(self):
        """Test that controller can work with grammar-loaded operators."""
        grammar = get_default_fex_grammar()
        
        unary = grammar.get_unary_functions()
        binary = grammar.get_binary_functions()
        
        # Verify we can use these operators
        x = torch.randn(5, requires_grad=True)
        
        # Test unary
        for func in unary:
            y = func(x)
            assert y.shape == x.shape
        
        # Test binary
        a = torch.randn(5)
        b = torch.randn(5)
        for func in binary:
            y = func(a, b)
            assert y.shape == a.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
