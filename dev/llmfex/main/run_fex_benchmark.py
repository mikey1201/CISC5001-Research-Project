#!/usr/bin/env python3
"""
FEX Benchmark Runner - Run FEX with different grammars on PDE problems

This script provides a practical interface to run FEX experiments locally
comparing default FEX grammar vs LLM-suggested grammars.

Usage:
    # Run quick test on first 3 problems (uses ground truth operators)
    python run_fex_benchmark.py --testset pde_testset_S101.jsonl --quick

    # Run with ACTUAL LLM predictions (connects your fine-tuned model!)
    python run_fex_benchmark.py --testset pde_testset_S101.jsonl --quick \
        --llm_model /path/to/your/finetuned/model

    # Run full benchmark
    python run_fex_benchmark.py --testset pde_testset_S101.jsonl --epochs 2000

    # Run specific problems
    python run_fex_benchmark.py --testset pde_testset_S101.jsonl --problems 0,1,2
"""

import json
import os
import sys
import subprocess
import argparse
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import math

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import from benchmark_pde
from benchmark_pde import (
    PDETestsetParser, 
    GrammarGenerator, 
    PDEProblem,
    BenchmarkResult
)


@dataclass
class FEXRunConfig:
    """Configuration for a FEX run."""
    dim: int = 3  # Number of spatial variables (will be set per-problem)
    epoch: int = 500
    tree: str = 'depth3'  # Tree depth (depth1, depth2, depth3)
    lr: float = 1e-2
    gpu: int = 0
    bs: int = 1
    checkpoint: str = 'checkpoints'
    
    # Grammar settings
    grammar_source: str = 'fex'
    grammar_path: str = ''
    
    # Problem config (NEW)
    problem_config_path: str = ''


class FEXBenchmarkRunner:
    """Run FEX benchmarks with different grammars."""
    
    def __init__(self, fex_root: str = None):
        """
        Initialize benchmark runner.
        
        Args:
            fex_root: Path to FEX root directory. If None, auto-detects.
        """
        # Try multiple possible locations for FEX
        possible_paths = [
            os.path.join(SCRIPT_DIR, 'fex', 'Poisson'),  # Same level as script
            os.path.join(SCRIPT_DIR, '..', 'fex', 'Poisson'),  # Parent level
            os.path.abspath(os.path.join(os.getcwd(), 'fex', 'Poisson')),  # CWD
            os.path.abspath(os.path.join(os.getcwd(), '..', 'fex', 'Poisson')),  # CWD parent
        ]
        
        if fex_root is not None:
            # User specified path - use it
            self.fex_root = fex_root
        else:
            # Auto-detect
            self.fex_root = None
            for path in possible_paths:
                controller = os.path.join(path, 'controller_poisson_grammar.py')
                if os.path.exists(controller):
                    self.fex_root = path
                    print(f"Found FEX at: {path}")
                    break
            
            if self.fex_root is None:
                # Default to first option (will show error later)
                self.fex_root = possible_paths[0]
                print(f"WARNING: FEX not found, using default path: {self.fex_root}")
        
        # Controller paths for different PDE types
        self.poisson_controller_path = os.path.join(self.fex_root, 'controller_poisson_grammar.py')
        # Conservation controller is in a different directory
        fex_base = os.path.dirname(self.fex_root)  # Go up from Poisson to fex/
        self.conservation_controller_path = os.path.join(fex_base, 'Conservationlaw', 'controller_conservative_grammar.py')
        
        # Default controller path (for backward compatibility)
        self.controller_path = self.poisson_controller_path
        
        self.parser = PDETestsetParser()
        self.grammar_gen = GrammarGenerator()
        
        # Results storage - use current working directory
        self.results_dir = os.path.join(os.getcwd(), 'benchmark_results')
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results directory: {self.results_dir}")
        print(f"Poisson controller: {self.poisson_controller_path}")
        print(f"Conservation controller: {self.conservation_controller_path}")
    
    def prepare_grammar_file(self, problem: PDEProblem, grammar_type: str) -> str:
        """Generate and save a grammar file for the problem."""
        if grammar_type == 'fex_default':
            grammar = self.grammar_gen.generate_fex_default_grammar()
        elif grammar_type == 'llm_suggested':
            grammar = self.grammar_gen.generate_llm_grammar(problem)
        else:
            raise ValueError(f"Unknown grammar type: {grammar_type}")
        
        grammar_path = os.path.join(
            self.results_dir,
            f"grammar_{problem.problem_id}_{grammar_type}.json"
        )
        self.grammar_gen.save_grammar(grammar, grammar_path)
        return grammar_path
    
    def prepare_problem_config(self, problem: PDEProblem) -> str:
        """Generate and save a problem config file for the problem."""
        config = {
            "problem_id": problem.problem_id,
            "pde_type": problem.pde_type,
            "rhs": problem.rhs,
            "solution": problem.solution,
            "boundary_conditions": problem.boundary_conditions,
            "variables": list(problem.variables) if problem.variables else ["x1", "x2", "x3"],
        }
        
        config_path = os.path.join(
            self.results_dir,
            f"problem_{problem.problem_id}_config.json"
        )
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def get_controller_path(self, pde_type: str) -> str:
        """
        Get the appropriate controller path based on PDE type.
        
        Args:
            pde_type: Type of PDE (Poisson, LinearConservation, etc.)
            
        Returns:
            Path to the appropriate controller script
        """
        pde_type_lower = pde_type.lower() if pde_type else 'poisson'
        
        if 'conservation' in pde_type_lower or 'linearconservation' in pde_type_lower:
            controller_path = self.conservation_controller_path
            if not os.path.exists(controller_path):
                print(f"WARNING: Conservation controller not found at {controller_path}")
                print(f"  Falling back to Poisson controller")
                controller_path = self.poisson_controller_path
        else:
            # Default to Poisson controller for Poisson and unknown types
            controller_path = self.poisson_controller_path
        
        return controller_path
    
    def build_command(self, config: FEXRunConfig, pde_type: str = 'Poisson') -> List[str]:
        """Build the FEX command line."""
        controller_path = self.get_controller_path(pde_type)
        cmd = [
            'python', controller_path,
            '--dim', str(config.dim),
            '--epoch', str(config.epoch),
            '--tree', config.tree,
            '--lr', str(config.lr),
            '--gpu', str(config.gpu),
            '--bs', str(config.bs),
            '--ckpt', config.checkpoint,
            '--grammar_source', config.grammar_source,
        ]
        
        if config.grammar_path:
            cmd.extend(['--grammar_path', config.grammar_path])
        
        # NEW: Add problem config
        if config.problem_config_path:
            cmd.extend(['--problem_config', config.problem_config_path])
        
        return cmd
    
    def run_fex(self, config: FEXRunConfig, timeout: int = 3600, verbose: bool = False, pde_type: str = 'Poisson') -> Dict[str, Any]:
        """
        Run FEX with the given configuration.
        
        Args:
            config: FEX run configuration
            timeout: Maximum execution time in seconds
            verbose: If True, show all training output
            pde_type: Type of PDE (Poisson, LinearConservation, etc.)
            
        Returns:
            Dictionary with run results
        """
        cmd = self.build_command(config, pde_type)
        controller_path = self.get_controller_path(pde_type)
        
        # Check if controller exists
        if not os.path.exists(controller_path):
            error_msg = f"Controller not found: {controller_path}"
            print(f"ERROR: {error_msg}")
            return {
                'command': ' '.join(cmd),
                'success': False,
                'output': '',
                'error': error_msg,
                'time_elapsed': 0,
            }
        
        # Check if grammar file exists
        if config.grammar_path and not os.path.exists(config.grammar_path):
            error_msg = f"Grammar file not found: {config.grammar_path}"
            print(f"ERROR: {error_msg}")
            return {
                'command': ' '.join(cmd),
                'success': False,
                'output': '',
                'error': error_msg,
                'time_elapsed': 0,
            }
        
        result = {
            'command': ' '.join(cmd),
            'success': False,
            'output': '',
            'error': '',
            'time_elapsed': 0,
            'pde_type': pde_type,
        }
        
        start_time = time.time()
        
        print(f"Running: {' '.join(cmd[:6])}... (epochs={config.epoch}, type={pde_type})")
        
        # Determine the working directory based on PDE type
        if 'conservation' in pde_type.lower():
            work_dir = os.path.dirname(controller_path)  # Conservationlaw/
        else:
            work_dir = self.fex_root  # Poisson/
        
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
            
            result['output'] = proc.stdout
            result['error'] = proc.stderr
            result['success'] = proc.returncode == 0
            result['returncode'] = proc.returncode
            
            # Parse output for metrics
            result.update(self._parse_fex_output(proc.stdout))
            
            # Show brief status
            elapsed = time.time() - start_time
            if result['success']:
                l2 = result.get('relative_l2_error', 'N/A')
                if isinstance(l2, float):
                    print(f"  Done in {elapsed:.1f}s | L2 error: {l2:.6f}")
                else:
                    print(f"  Done in {elapsed:.1f}s")
            else:
                print(f"  FAILED (code {proc.returncode})")
                if verbose and proc.stderr:
                    print(f"  Error: {proc.stderr[:200]}")
            
        except subprocess.TimeoutExpired:
            result['error'] = f"Process timed out after {timeout} seconds"
            print(f"  TIMEOUT after {timeout} seconds")
        except Exception as e:
            result['error'] = str(e)
            print(f"  EXCEPTION: {e}")
        
        result['time_elapsed'] = time.time() - start_time
        
        return result
    
    def _parse_fex_output(self, output: str) -> Dict[str, Any]:
        """Parse FEX output for metrics."""
        metrics = {}
        
        # Look for relative L2 error (supports scientific notation)
        l2_match = re.search(r'relative l2 error:\s*([\d.e+-]+)', output)
        if l2_match:
            try:
                metrics['relative_l2_error'] = float(l2_match.group(1))
            except ValueError:
                pass
        
        # Look for final loss/error
        error_matches = re.findall(r'error[:\s]+([\d.e+-]+)', output.lower())
        if error_matches:
            try:
                metrics['final_error'] = float(error_matches[-1])
            except:
                pass
        
        # Look for best error
        best_match = re.search(r'min:\s*([\d.e+-]+)', output)
        if best_match:
            try:
                metrics['best_error'] = float(best_match.group(1))
            except:
                pass
        
        # Look for formulas
        formula_matches = re.findall(r'Formula[:\s]+(.+)', output)
        if formula_matches:
            metrics['found_formulas'] = formula_matches[-5:]  # Last 5 formulas
        
        return metrics
    
    def run_benchmark(self, testset_path: str, epochs: int = 500,
                      quick_mode: bool = False,
                      problem_ids: Optional[List[int]] = None,
                      skip_fex_default: bool = False,
                      skip_llm: bool = False,
                      tree_name: str = 'depth3') -> List[Dict]:
        """
        Run the full benchmark.
        
        Args:
            testset_path: Path to the test set JSONL file
            epochs: Number of training epochs
            quick_mode: If True, run only first 3 problems with fewer epochs
            problem_ids: Specific problem IDs to run (None for all)
            skip_fex_default: Skip running FEX with default grammar
            skip_llm: Skip running FEX with LLM grammar
            tree_name: Tree depth name ('depth1', 'depth2', 'depth3')
            
        Returns:
            List of result dictionaries
        """
        problems = self.parser.parse_file(testset_path)
        
        if problem_ids:
            problems = [p for p in problems if p.problem_id in problem_ids]
        
        if quick_mode:
            problems = problems[:3]
            epochs = min(epochs, 100)
        
        all_results = []
        
        for problem in problems:
            # Determine dim from problem variables
            problem_dim = len(problem.variables) if problem.variables else 3
            
            print(f"\n{'='*70}")
            print(f"Problem {problem.problem_id}: {problem.pde_type}")
            print(f"Target: {problem.solution[:60]}...")
            print(f"Operators: {problem.unary_ops} / {problem.binary_ops}")
            print(f"Variables: {problem.variables} (dim={problem_dim})")
            print(f"Tree: {tree_name}")
            print(f"{'='*70}")
            
            # Create checkpoint directory for this problem
            ckpt_dir = os.path.join(
                self.results_dir, 
                f"ckpt_p{problem.problem_id}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Generate problem config (used by both runs)
            problem_config_path = self.prepare_problem_config(problem)
            print(f"Problem config: {problem_config_path}")
            
            # Run with FEX default grammar
            if not skip_fex_default:
                print("\n--- FEX Default Grammar ---")
                grammar_path = self.prepare_grammar_file(problem, 'fex_default')
                
                config = FEXRunConfig(
                    dim=problem_dim,
                    tree=tree_name,
                    epoch=epochs,
                    checkpoint=ckpt_dir + '_default',
                    grammar_source='llm',  # Use llm source with generated file
                    grammar_path=grammar_path,
                    problem_config_path=problem_config_path,
                )
                
                result = self.run_fex(config, pde_type=problem.pde_type)
                result['problem_id'] = problem.problem_id
                result['pde_type'] = problem.pde_type
                result['grammar_type'] = 'fex_default'
                result['target_formula'] = problem.solution
                all_results.append(result)
            
            # Run with LLM-suggested grammar
            if not skip_llm:
                print("\n--- LLM-Suggested Grammar ---")
                grammar_path = self.prepare_grammar_file(problem, 'llm_suggested')
                
                config = FEXRunConfig(
                    dim=problem_dim,
                    tree=tree_name,
                    epoch=epochs,
                    checkpoint=ckpt_dir + '_llm',
                    grammar_source='llm',
                    grammar_path=grammar_path,
                    problem_config_path=problem_config_path,
                )
                
                result = self.run_fex(config, pde_type=problem.pde_type)
                result['problem_id'] = problem.problem_id
                result['pde_type'] = problem.pde_type
                result['grammar_type'] = 'llm_suggested'
                result['target_formula'] = problem.solution
                all_results.append(result)
        
        return all_results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    
    def generate_summary(self, results: List[Dict]) -> str:
        """Generate a summary report."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("BENCHMARK RESULTS SUMMARY")
        lines.append("=" * 70)
        
        # Group by grammar type
        by_grammar = defaultdict(list)
        for r in results:
            by_grammar[r.get('grammar_type', 'unknown')].append(r)
        
        # Per-problem comparison table
        lines.append("\nPer-Problem Results:")
        lines.append("-" * 70)
        lines.append(f"{'Problem':<10} {'Grammar':<18} {'Success':<10} {'L2 Error':<12} {'Time (s)':<10}")
        lines.append("-" * 70)
        
        for result in results:
            pid = result.get('problem_id', '?')
            gtype = result.get('grammar_type', '?')
            success = "Yes" if result.get('success') else "No"
            l2 = result.get('relative_l2_error')
            l2_str = f"{l2:.6f}" if l2 else "N/A"
            time_str = f"{result.get('time_elapsed', 0):.1f}"
            lines.append(f"{pid:<10} {gtype:<18} {success:<10} {l2_str:<12} {time_str:<10}")
        
        # Aggregate statistics
        lines.append("\n" + "=" * 70)
        lines.append("Aggregate Statistics:")
        lines.append("-" * 70)
        
        for grammar_type, grammar_results in by_grammar.items():
            successes = [r for r in grammar_results if r.get('success')]
            l2_errors = [r.get('relative_l2_error') for r in successes if r.get('relative_l2_error')]
            times = [r.get('time_elapsed') for r in grammar_results if r.get('time_elapsed')]
            
            lines.append(f"\n{grammar_type.upper()}:")
            lines.append(f"  Successful runs: {len(successes)}/{len(grammar_results)}")
            
            if l2_errors:
                avg_l2 = sum(l2_errors) / len(l2_errors)
                min_l2 = min(l2_errors)
                max_l2 = max(l2_errors)
                lines.append(f"  L2 Error - Avg: {avg_l2:.6f}, Min: {min_l2:.6f}, Max: {max_l2:.6f}")
            
            if times:
                avg_time = sum(times) / len(times)
                lines.append(f"  Time - Avg: {avg_time:.1f}s, Total: {sum(times):.1f}s")
        
        # Comparison
        if 'fex_default' in by_grammar and 'llm_suggested' in by_grammar:
            lines.append("\n" + "=" * 70)
            lines.append("COMPARISON: LLM Grammar vs FEX Default")
            lines.append("-" * 70)
            
            # Match problems
            default_results = {r['problem_id']: r for r in by_grammar['fex_default']}
            llm_results = {r['problem_id']: r for r in by_grammar['llm_suggested']}
            
            improvements = []
            for pid in default_results:
                if pid in llm_results:
                    d_l2 = default_results[pid].get('relative_l2_error')
                    l_l2 = llm_results[pid].get('relative_l2_error')
                    if d_l2 and l_l2 and d_l2 > 0:
                        imp = (d_l2 - l_l2) / d_l2 * 100
                        improvements.append((pid, d_l2, l_l2, imp))
            
            if improvements:
                lines.append(f"\n{'Problem':<10} {'Default L2':<12} {'LLM L2':<12} {'Improvement':<12}")
                lines.append("-" * 50)
                for pid, d_l2, l_l2, imp in improvements:
                    sign = "+" if imp > 0 else ""
                    lines.append(f"{pid:<10} {d_l2:<12.6f} {l_l2:<12.6f} {sign}{imp:.1f}%")
                
                avg_imp = sum(i[3] for i in improvements) / len(improvements)
                better_count = sum(1 for i in improvements if i[3] > 0)
                
                lines.append("-" * 50)
                lines.append(f"Average improvement: {avg_imp:+.1f}%")
                lines.append(f"LLM better on: {better_count}/{len(improvements)} problems")
                
                if avg_imp > 0:
                    lines.append("\n>>> LLM-ENHANCED FEX PERFORMS BETTER <<<")
                else:
                    lines.append("\n>>> DEFAULT FEX PERFORMS BETTER <<<")
        
        return "\n".join(lines)


def test_single_run(grammar_path: str, epochs: int = 10, gpu: int = 0, dim: int = 3, tree: str = 'depth3'):
    """Test a single FEX run directly - shows all output for debugging."""
    runner = FEXBenchmarkRunner()
    
    print("=" * 60)
    print("DIRECT TEST MODE - Running FEX directly")
    print("=" * 60)
    print(f"Controller: {runner.controller_path}")
    print(f"Grammar: {grammar_path}")
    print(f"Epochs: {epochs}")
    print(f"Dim: {dim}")
    print(f"Tree: {tree}")
    print(f"GPU: {gpu}")
    print("=" * 60)
    
    if not os.path.exists(runner.controller_path):
        print(f"ERROR: Controller not found!")
        print(f"Searched at: {runner.controller_path}")
        return None
    
    if not os.path.exists(grammar_path):
        print(f"ERROR: Grammar file not found!")
        print(f"Searched at: {grammar_path}")
        return None
    
    # Run capturing output, then parse and display summary
    import subprocess
    cmd = [
        'python', runner.controller_path,
        '--dim', str(dim),
        '--epoch', str(epochs),
        '--tree', tree,
        '--lr', '0.01',
        '--gpu', str(gpu),
        '--bs', '1',
        '--ckpt', 'test_checkpoint',
        '--grammar_source', 'llm',
        '--grammar_path', grammar_path,
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("(Training in progress... output suppressed)\n")
    
    # Run with output capture
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=runner.fex_root)
    
    # Parse output for key metrics
    output = result.stdout + result.stderr
    metrics = runner._parse_fex_output(output)
    
    # Print summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Return code: {result.returncode}")
    print(f"Success: {result.returncode == 0}")
    
    if metrics.get('relative_l2_error'):
        print(f"Relative L2 Error: {metrics['relative_l2_error']:.6f}")
    if metrics.get('final_error'):
        print(f"Final Error: {metrics['final_error']:.6e}")
    if metrics.get('best_error'):
        print(f"Best Error: {metrics['best_error']:.6e}")
    
    if metrics.get('found_formulas'):
        print(f"\nFound Formulas (last 3):")
        for f in metrics['found_formulas'][-3:]:
            print(f"  {f}")
    
    # Show any errors
    if result.returncode != 0:
        print("\nERRORS:")
        if result.stderr:
            print(result.stderr[-500:])
        if result.stdout:
            print(result.stdout[-500:])
    
    print("=" * 60)
    
    return {
        'success': result.returncode == 0,
        'relative_l2_error': metrics.get('relative_l2_error'),
        'final_error': metrics.get('final_error'),
        'best_error': metrics.get('best_error'),
        'formulas': metrics.get('found_formulas', []),
    }


def main():
    parser = argparse.ArgumentParser(description='Run FEX benchmarks with different grammars')
    parser.add_argument('--testset', type=str, required=False,
                        help='Path to PDE testset file (JSONL)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: test only 3 problems with 100 epochs')
    parser.add_argument('--problems', type=str, default=None,
                        help='Comma-separated problem IDs to test')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')
    parser.add_argument('--skip-default', action='store_true',
                        help='Skip FEX default grammar runs')
    parser.add_argument('--skip-llm', action='store_true',
                        help='Skip LLM grammar runs')
    parser.add_argument('--depth', type=int, default=3,
                        help='Tree depth (1, 2, or 3). Default: 3')
    
    # LLM integration
    parser.add_argument('--llm_model', type=str, default=None,
                        help='Path to fine-tuned LLM (uses ACTUAL LLM predictions!)')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Base model for LLM')
    
    # Test mode
    parser.add_argument('--test', action='store_true',
                        help='Test mode: run single problem directly with visible output')
    parser.add_argument('--grammar', type=str, default=None,
                        help='Path to grammar file (for --test mode)')
    
    args = parser.parse_args()
    
    # Map depth to FEX tree name
    depth_to_tree = {1: 'depth1', 2: 'depth2', 3: 'depth3'}
    tree_name = depth_to_tree.get(args.depth, 'depth3')
    
    # Test mode
    if args.test:
        if args.grammar:
            test_single_run(args.grammar, args.epochs, args.gpu, dim=3, tree=tree_name)
        else:
            # Generate a test grammar and run
            from benchmark_pde import PDEProblem
            runner = FEXBenchmarkRunner()
            
            # Create a minimal test problem
            test_problem = PDEProblem(
                problem_id=0,
                pde_type='Poisson',
                rhs='const x2 *',
                boundary_conditions=[],
                solution='const x2 *',
                raw_prompt='',
                unary_ops={'identity'},
                binary_ops={'mul'}
            )
            
            grammar_path = runner.prepare_grammar_file(test_problem, 'llm_suggested')
            print(f"Generated test grammar: {grammar_path}")
            
            test_single_run(grammar_path, epochs=args.epochs or 10, gpu=args.gpu, dim=3, tree=tree_name)
        return
    
    if not args.testset:
        parser.error('--testset is required unless using --test mode')
    
    # Parse problem IDs
    problem_ids = None
    if args.problems:
        problem_ids = [int(x.strip()) for x in args.problems.split(',')]
    
    # Run benchmark
    runner = FEXBenchmarkRunner()
    
    # If LLM model is specified, use actual LLM predictions
    if args.llm_model:
        print("\n" + "=" * 60)
        print("USING ACTUAL LLM FOR GRAMMAR PREDICTIONS")
        print(f"Model: {args.llm_model}")
        print("=" * 60 + "\n")
        
        # Import LLM bridge
        try:
            from llm_to_fex_bridge import LLMInference, create_fex_grammar_from_operators, classify_operators
            from benchmark_pde import PDETestsetParser
        except ImportError as e:
            print(f"ERROR: Could not import LLM bridge: {e}")
            print("Make sure llm_to_fex_bridge.py is in the same directory")
            return
        
        # Load LLM
        llm = LLMInference(args.llm_model, args.base_model)
        llm.load()
        
        # Parse problems
        parser_obj = PDETestsetParser()
        problems = parser_obj.parse_file(args.testset)
        
        if problem_ids:
            problems = [p for p in problems if p.problem_id in problem_ids]
        
        if args.quick:
            problems = problems[:3]
            args.epochs = min(args.epochs, 100)
        
        all_results = []
        
        for problem in problems:
            print(f"\n{'='*70}")
            print(f"Problem {problem.problem_id}: {problem.pde_type}")
            print(f"Target: {problem.solution[:60]}...")
            print(f"{'='*70}")
            
            # Determine dim from problem variables
            problem_dim = len(problem.variables) if problem.variables else 3
            
            # Get LLM prediction
            print("\n--- Getting LLM Prediction ---")
            unary_ops, binary_ops, prediction = llm.predict_operators(problem.raw_prompt)
            
            print(f"LLM predicted: {prediction[:60]}...")
            print(f"Extracted unary: {unary_ops}")
            print(f"Extracted binary: {binary_ops}")
            
            # Create grammar from LLM prediction
            grammar = create_fex_grammar_from_operators(
                unary_ops, binary_ops, 
                problem_id=problem.problem_id,
                pde_type=problem.pde_type
            )
            
            # Save grammar
            grammar_path = os.path.join(
                runner.results_dir,
                f"grammar_llm_{problem.problem_id}.json"
            )
            with open(grammar_path, 'w') as f:
                json.dump(grammar, f, indent=2)
            
            # Run FEX with LLM-predicted grammar
            print("\n--- Running FEX with LLM Grammar ---")
            ckpt_dir = os.path.join(runner.results_dir, f"ckpt_p{problem.problem_id}_llm")
            
            config = FEXRunConfig(
                dim=problem_dim,
                tree=tree_name,
                epoch=args.epochs,
                checkpoint=ckpt_dir,
                grammar_source='llm',
                grammar_path=grammar_path,
            )
            
            result = runner.run_fex(config)
            result['problem_id'] = problem.problem_id
            result['pde_type'] = problem.pde_type
            result['grammar_type'] = 'llm_predicted'
            result['target_formula'] = problem.solution
            result['llm_prediction'] = prediction
            result['llm_unary_ops'] = list(unary_ops)
            result['llm_binary_ops'] = list(binary_ops)
            all_results.append(result)
        
        results = all_results
    else:
        # Original behavior - extract operators from ground truth
        results = runner.run_benchmark(
            args.testset,
            epochs=args.epochs,
            quick_mode=args.quick,
            problem_ids=problem_ids,
            skip_fex_default=args.skip_default,
            skip_llm=args.skip_llm,
            tree_name=tree_name
        )
    
    # Save results
    output_path = os.path.join(runner.results_dir, args.output)
    runner.save_results(results, output_path)
    
    # Print summary
    summary = runner.generate_summary(results)
    print("\n" + summary)


if __name__ == '__main__':
    main()
