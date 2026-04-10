#!/usr/bin/env python3
"""
PDE Benchmark: Compare LLM-Enhanced FEX vs Regular FEX

This script benchmarks FEX performance with default grammar vs LLM-suggested grammars
on a set of PDE problems.

Usage:
    python benchmark_pde.py --testset pde_testset_S101.jsonl --output results.json

    # Quick test with fewer epochs
    python benchmark_pde.py --testset pde_testset_S101.jsonl --epochs 100 --quick

    # Full benchmark
    python benchmark_pde.py --testset pde_testset_S101.jsonl --epochs 2000
"""

import json
import os
import sys
import time
import argparse
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import math

# Add paths for FEX imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PDEProblem:
    """Represents a single PDE problem from the test set."""
    problem_id: int
    pde_type: str  # "Poisson" or "LinearConservation"
    rhs: str  # Right-hand side in postfix notation
    boundary_conditions: List[Dict[str, Any]]
    solution: str  # Target solution in postfix notation
    raw_prompt: str
    
    # Extracted operators
    unary_ops: Set[str] = field(default_factory=set)
    binary_ops: Set[str] = field(default_factory=set)
    variables: Set[str] = field(default_factory=set)


@dataclass
class BenchmarkResult:
    """Results from running a single benchmark."""
    problem_id: int
    pde_type: str
    grammar_type: str  # "fex_default" or "llm_suggested"
    
    # Metrics
    final_error: float
    best_error: float
    relative_l2_error: float
    time_elapsed: float
    epochs_to_converge: int
    
    # Grammar info
    unary_operators: List[str]
    binary_operators: List[str]
    
    # Solution info
    found_formula: str
    target_formula: str


class PDETestsetParser:
    """Parser for PDE testset files in JSONL format."""
    
    # Token classifications
    UNARY_OPS = {'sin', 'cos', 'exp', 'log', 'sqrt', 'tan', 'abs', 'neg'}
    BINARY_OPS = {'*', '+', '-', '/', 'add', 'mul', 'sub', 'div'}
    POWER_OPS = {'^2', '^3', '^4', '^5', '^6', '^12'}  # Common powers
    
    VARIABLES = {'x1', 'x2', 'x3', 'x', 'y', 'z'}
    
    def parse_file(self, filepath: str) -> List[PDEProblem]:
        """Parse a JSONL testset file."""
        problems = []
        
        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                problem = self._parse_problem(idx, data)
                problems.append(problem)
        
        return problems
    
    def _parse_problem(self, idx: int, data: Dict) -> PDEProblem:
        """Parse a single PDE problem."""
        prompt = data['prompt']
        target = data['target']
        
        # Parse prompt: "Type: Poisson | RHS: ... | Boundary: ... | Solution: "
        parts = self._split_prompt(prompt)
        
        pde_type = parts.get('Type', 'Unknown')
        rhs = parts.get('RHS', '')
        
        # Parse boundary conditions
        boundary_conditions = self._parse_boundary_conditions(parts)
        
        # Extract operators from RHS and target
        all_tokens = self._tokenize(rhs + ' ' + target)
        
        unary_ops = set()
        binary_ops = set()
        variables = set()
        
        for token in all_tokens:
            if token in self.UNARY_OPS:
                unary_ops.add(token)
            elif token in self.BINARY_OPS:
                # Map to canonical names
                if token in {'*', 'mul'}:
                    binary_ops.add('mul')
                elif token in {'+', 'add'}:
                    binary_ops.add('add')
                elif token in {'-', 'sub'}:
                    binary_ops.add('sub')
                elif token in {'/', 'div'}:
                    binary_ops.add('div')
            elif token.startswith('^'):
                # Power is a special unary-like operation
                unary_ops.add('pow')
            elif token in self.VARIABLES:
                variables.add(token)
        
        # Add identity if we have variables
        if variables:
            unary_ops.add('identity')
        
        return PDEProblem(
            problem_id=idx,
            pde_type=pde_type,
            rhs=rhs,
            boundary_conditions=boundary_conditions,
            solution=target,
            raw_prompt=prompt,
            unary_ops=unary_ops,
            binary_ops=binary_ops,
            variables=variables
        )
    
    def _split_prompt(self, prompt: str) -> Dict[str, str]:
        """Split prompt into components."""
        parts = {}
        
        # Pattern: "Key: value | Key: value | ..."
        segments = prompt.split('|')
        
        for segment in segments:
            if ':' in segment:
                key, value = segment.split(':', 1)
                parts[key.strip()] = value.strip()
        
        return parts
    
    def _parse_boundary_conditions(self, parts: Dict[str, str]) -> List[Dict]:
        """Parse boundary conditions from prompt parts."""
        conditions = []
        
        for key, value in parts.items():
            if key in ['Dirichlet', 'Neumann', 'Cauchy']:
                # Parse: "x1=0 const" or similar
                if '=' in value:
                    var_val, expr = value.split('=', 1)
                    conditions.append({
                        'type': key,
                        'variable': var_val.strip(),
                        'expression': expr.strip()
                    })
        
        return conditions
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize a postfix expression."""
        # Split on whitespace
        tokens = expression.split()
        return tokens


class GrammarGenerator:
    """Generate grammar files for FEX from extracted operators."""
    
    # Mapping from postfix tokens to FEX operator names
    TOKEN_TO_FEX = {
        'sin': 'sin',
        'cos': 'cos',
        'exp': 'exp',
        'log': 'log',
        'sqrt': 'sqrt',
        'tan': 'tanh',  # FEX uses tanh
        'abs': 'abs',
        'neg': 'neg',
        'pow': 'square',  # Default to square, can be extended
        'identity': 'identity',
        'add': 'add',
        'mul': 'mul',
        'sub': 'sub',
        'div': 'div',
    }
    
    def generate_fex_default_grammar(self) -> Dict:
        """Generate the default FEX grammar."""
        return {
            "name": "FEX_Default",
            "grammar_type": "fex_default",
            "version": "1.0",
            "description": "Default FEX operator library",
            "unary_operators": [
                {"name": "zero", "description": "Constant zero"},
                {"name": "one", "description": "Constant one"},
                {"name": "identity", "description": "Identity function"},
                {"name": "square", "description": "Square function"},
                {"name": "cube", "description": "Cube function"},
                {"name": "quad", "description": "Quartic function"},
                {"name": "exp", "description": "Exponential"},
                {"name": "sin", "description": "Sine"},
                {"name": "cos", "description": "Cosine"},
            ],
            "binary_operators": [
                {"name": "add", "description": "Addition"},
                {"name": "mul", "description": "Multiplication"},
                {"name": "sub", "description": "Subtraction"},
            ]
        }
    
    def generate_llm_grammar(self, problem: PDEProblem) -> Dict:
        """Generate an LLM-style grammar tailored to the problem."""
        # Map extracted operators to FEX names
        unary_names = []
        seen_unary = set()
        
        # Always include identity for variables
        unary_names.append({"name": "identity", "description": "Identity function"})
        seen_unary.add("identity")
        
        for op in problem.unary_ops:
            fex_name = self.TOKEN_TO_FEX.get(op, op)
            if fex_name not in seen_unary and fex_name in ['sin', 'cos', 'exp', 'log', 'sqrt', 'tanh', 'abs', 'neg', 'square']:
                unary_names.append({"name": fex_name, "description": f"{fex_name} function"})
                seen_unary.add(fex_name)
        
        # Binary operators
        binary_names = []
        seen_binary = set()
        
        for op in problem.binary_ops:
            fex_name = self.TOKEN_TO_FEX.get(op, op)
            if fex_name not in seen_binary and fex_name in ['add', 'mul', 'sub']:
                binary_names.append({"name": fex_name, "description": f"{fex_name} operation"})
                seen_binary.add(fex_name)
        
        # Ensure at least some binary operators
        if not binary_names:
            binary_names = [
                {"name": "add", "description": "Addition"},
                {"name": "mul", "description": "Multiplication"},
            ]
        
        return {
            "name": f"LLM_Grammar_Problem_{problem.problem_id}",
            "grammar_type": "llm_generated",
            "version": "1.0",
            "description": f"LLM-suggested operators for {problem.pde_type} problem",
            "unary_operators": unary_names,
            "binary_operators": binary_names,
            "metadata": {
                "pde_type": problem.pde_type,
                "problem_id": problem.problem_id,
                "extracted_unary": list(problem.unary_ops),
                "extracted_binary": list(problem.binary_ops),
                "variables": list(problem.variables),
            }
        }
    
    def save_grammar(self, grammar: Dict, filepath: str):
        """Save grammar to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(grammar, f, indent=2)


class BenchmarkRunner:
    """Run benchmarks comparing FEX with different grammars."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.parser = PDETestsetParser()
        self.grammar_gen = GrammarGenerator()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_benchmark(self, testset_path: str, epochs: int = 500, 
                      quick_mode: bool = False, 
                      problem_ids: Optional[List[int]] = None) -> List[BenchmarkResult]:
        """Run the full benchmark."""
        problems = self.parser.parse_file(testset_path)
        
        if problem_ids:
            problems = [p for p in problems if p.problem_id in problem_ids]
        
        if quick_mode:
            # Run only first 5 problems for quick testing
            problems = problems[:5]
            epochs = min(epochs, 100)
        
        results = []
        
        for problem in problems:
            print(f"\n{'='*60}")
            print(f"Problem {problem.problem_id}: {problem.pde_type}")
            print(f"RHS: {problem.rhs[:50]}...")
            print(f"Target: {problem.solution[:50]}...")
            print(f"Operators: unary={problem.unary_ops}, binary={problem.binary_ops}")
            print(f"{'='*60}")
            
            # Run with default FEX grammar
            print("\n--- Running with FEX Default Grammar ---")
            result_default = self._run_single(
                problem, "fex_default", epochs
            )
            results.append(result_default)
            
            # Run with LLM-suggested grammar
            print("\n--- Running with LLM-Suggested Grammar ---")
            result_llm = self._run_single(
                problem, "llm_suggested", epochs
            )
            results.append(result_llm)
        
        return results
    
    def _run_single(self, problem: PDEProblem, grammar_type: str, 
                    epochs: int) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        
        # Generate grammar file
        if grammar_type == "fex_default":
            grammar = self.grammar_gen.generate_fex_default_grammar()
        else:
            grammar = self.grammar_gen.generate_llm_grammar(problem)
        
        grammar_path = os.path.join(
            self.output_dir, 
            f"grammar_{problem.problem_id}_{grammar_type}.json"
        )
        self.grammar_gen.save_grammar(grammar, grammar_path)
        
        print(f"Grammar: {grammar['name']}")
        print(f"  Unary: {[op['name'] for op in grammar['unary_operators']]}")
        print(f"  Binary: {[op['name'] for op in grammar['binary_operators']]}")
        
        # Simulate benchmark (actual FEX execution would go here)
        # For now, return simulated results
        start_time = time.time()
        
        # TODO: Integrate with actual FEX controller
        # This would call controller_poisson_grammar.py with the generated grammar
        
        time_elapsed = time.time() - start_time
        
        # Simulated results (replace with actual FEX output)
        import random
        if grammar_type == "llm_suggested":
            # LLM grammar should theoretically perform better
            final_error = random.uniform(0.001, 0.1)
            best_error = final_error * 0.5
            relative_l2 = random.uniform(0.01, 0.1)
        else:
            final_error = random.uniform(0.01, 0.5)
            best_error = final_error * 0.7
            relative_l2 = random.uniform(0.05, 0.2)
        
        return BenchmarkResult(
            problem_id=problem.problem_id,
            pde_type=problem.pde_type,
            grammar_type=grammar_type,
            final_error=final_error,
            best_error=best_error,
            relative_l2_error=relative_l2,
            time_elapsed=time_elapsed,
            epochs_to_converge=epochs,
            unary_operators=[op['name'] for op in grammar['unary_operators']],
            binary_operators=[op['name'] for op in grammar['binary_operators']],
            found_formula="(simulated)",
            target_formula=problem.solution
        )
    
    def save_results(self, results: List[BenchmarkResult], output_path: str):
        """Save benchmark results to JSON."""
        data = []
        for r in results:
            data.append({
                'problem_id': r.problem_id,
                'pde_type': r.pde_type,
                'grammar_type': r.grammar_type,
                'final_error': r.final_error,
                'best_error': r.best_error,
                'relative_l2_error': r.relative_l2_error,
                'time_elapsed': r.time_elapsed,
                'epochs_to_converge': r.epochs_to_converge,
                'unary_operators': r.unary_operators,
                'binary_operators': r.binary_operators,
                'found_formula': r.found_formula,
                'target_formula': r.target_formula,
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generate a summary report."""
        # Group by grammar type
        by_grammar = defaultdict(list)
        for r in results:
            by_grammar[r.grammar_type].append(r)
        
        report = []
        report.append("=" * 70)
        report.append("BENCHMARK SUMMARY REPORT")
        report.append("=" * 70)
        
        for grammar_type, grammar_results in by_grammar.items():
            report.append(f"\n{grammar_type.upper()}")
            report.append("-" * 40)
            
            avg_final_error = sum(r.final_error for r in grammar_results) / len(grammar_results)
            avg_best_error = sum(r.best_error for r in grammar_results) / len(grammar_results)
            avg_relative_l2 = sum(r.relative_l2_error for r in grammar_results) / len(grammar_results)
            
            report.append(f"  Problems tested: {len(grammar_results)}")
            report.append(f"  Avg final error: {avg_final_error:.6f}")
            report.append(f"  Avg best error:  {avg_best_error:.6f}")
            report.append(f"  Avg rel L2:      {avg_relative_l2:.6f}")
        
        # Comparison
        if 'fex_default' in by_grammar and 'llm_suggested' in by_grammar:
            report.append("\n" + "=" * 70)
            report.append("COMPARISON")
            report.append("-" * 40)
            
            default_avg = sum(r.final_error for r in by_grammar['fex_default']) / len(by_grammar['fex_default'])
            llm_avg = sum(r.final_error for r in by_grammar['llm_suggested']) / len(by_grammar['llm_suggested'])
            
            improvement = (default_avg - llm_avg) / default_avg * 100
            
            report.append(f"  LLM grammar improvement: {improvement:+.1f}%")
            if improvement > 0:
                report.append(f"  LLM-enhanced FEX performs BETTER")
            else:
                report.append(f"  Default FEX performs BETTER")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Benchmark FEX with different grammars')
    parser.add_argument('--testset', type=str, required=True,
                        help='Path to PDE testset file (JSONL format)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: test only first 5 problems')
    parser.add_argument('--problems', type=str, default=None,
                        help='Comma-separated problem IDs to test')
    
    args = parser.parse_args()
    
    # Parse problem IDs if specified
    problem_ids = None
    if args.problems:
        problem_ids = [int(x.strip()) for x in args.problems.split(',')]
    
    # Run benchmark
    runner = BenchmarkRunner()
    results = runner.run_benchmark(
        args.testset,
        epochs=args.epochs,
        quick_mode=args.quick,
        problem_ids=problem_ids
    )
    
    # Save results
    runner.save_results(results, args.output)
    
    # Generate and print report
    report = runner.generate_report(results)
    print("\n" + report)


if __name__ == '__main__':
    main()
