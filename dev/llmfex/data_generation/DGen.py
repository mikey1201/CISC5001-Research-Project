"""
DGen.py - Main Data Generation Pipeline

Generates training dataset for LLM fine-tuning following paper methodology.

Paper Section 5.1 Experimental Setup:
- 198,000 total equations (99,000 per PDE type)
- PDE Types: Poisson, LinearConservationLaw
- Boundary Conditions: Cauchy, Dirichlet, Neumann
- Tree Depth: 3
- Variables: x0, x1 (2D default)

Output Format (Figure 3):
"Type: <PDE> | RHS: <f_ops> | <BC_Type>: <boundary> <g_ops> | Solution: || <u_ops>"
"""

import json
import random
import os
import sys
import gc
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

# Import local modules
from REGen import RandomExpressionGenerator
from Deriver import Deriver, BoundaryFace, create_boundary_face
from RPNC import assemble_data_point


# ==========================================================
# CONFIGURATION (Paper Section 5.1)
# ==========================================================

@dataclass
class DataGenConfig:
    """Configuration for data generation matching paper setup."""
    
    # Paper Section 5.1: "Using a tree depth of 3"
    TREE_DEPTH: int = 3
    
    # Number of variables (paper uses up to 4 based on Table 4, 5 examples)
    NUM_VARS: int = 2
    
    # Paper Section 5.1: 99,000 examples for each PDE type
    # Total: 198,000
    SAMPLES_PER_PDE_TYPE: int = 99000
    
    # PDE Types
    PDE_TYPES: List[str] = field(default_factory=lambda: ['Poisson', 'LinearConservationLaw'])
    
    # Boundary Condition Types
    BC_TYPES: List[str] = field(default_factory=lambda: ['Cauchy', 'Dirichlet', 'Neumann'])
    
    # Domain bounds (paper doesn't specify, using reasonable defaults)
    DOMAIN_BOUNDS: Tuple[float, float] = (0.0, 1.0)
    
    # Boundary values to use
    BOUNDARY_VALUES: List[float] = field(default_factory=lambda: [0.0, 1.0])
    
    # Output settings
    OUTPUT_DIR: str = "./data"
    OUTPUT_FILENAME: str = "pde_dataset.json"
    
    # Random seed for reproducibility
    SEED: Optional[int] = 42
    
    # Batch save interval (save every N samples to prevent data loss)
    SAVE_INTERVAL: int = 1000


def generate_single_sample(
    expr_generator: RandomExpressionGenerator,
    deriver: PDEDeriver,
    pde_type: str,
    bc_type: str,
    config: DataGenConfig,
    max_retries: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Generate a single training sample.
    
    Args:
        expr_generator: Expression generator instance
        deriver: PDE deriver instance
        pde_type: Type of PDE
        bc_type: Type of boundary condition
        config: Configuration object
        max_retries: Maximum retry attempts for generation
        
    Returns:
        Assembled data point dictionary, or None if failed
    """
    for attempt in range(max_retries):
        try:
            # Step 1-2: Generate random solution u(x)
            u_expr, meta = expr_generator.generate_u()
            
            # Step 3: Compute f(x) from PDE
            f_expr = deriver.calculate_f(u_expr, pde_type)
            
            # Step 4: Select boundary face and compute g(x)
            boundary_face = create_boundary_face(
                expr_generator.vars,
                value=random.choice(config.BOUNDARY_VALUES)
            )
            
            # Handle Cauchy separately - need TWO components
            if bc_type == 'Cauchy':
                g_value, g_flux = deriver.calculate_g_cauchy(u_expr, boundary_face)
                
                # Validate expressions are not too complex
                if len(str(f_expr)) > 1000 or len(str(g_value)) > 1000 or len(str(g_flux)) > 1000:
                    continue
                
                # Assemble with both Cauchy components
                data_point = assemble_data_point(
                    pde_type=pde_type,
                    bc_type=bc_type,
                    boundary_face=boundary_face,
                    f_expr=f_expr,
                    g_expr=g_value,  # Function value component
                    u_expr=u_expr,
                    g_flux_expr=g_flux  # Normal derivative component
                )
            else:
                # Standard Dirichlet/Neumann case
                g_expr = deriver.calculate_g(u_expr, bc_type, boundary_face)
                
                # Validate expressions are not too complex
                if len(str(f_expr)) > 1000 or len(str(g_expr)) > 1000:
                    continue
                
                # Assemble into training format
                data_point = assemble_data_point(
                    pde_type=pde_type,
                    bc_type=bc_type,
                    boundary_face=boundary_face,
                    f_expr=f_expr,
                    g_expr=g_expr,
                    u_expr=u_expr
                )
            
            return data_point
            
        except Exception as e:
            continue
    
    return None


def generate_dataset(
    config: DataGenConfig,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate the complete dataset.
    
    Paper Section 5.1:
    "we randomly generate a dataset of 198,000 equations, with 99,000 
    examples for each PDE type"
    """
    # Set random seed
    if config.SEED is not None:
        random.seed(config.SEED)
    
    # Initialize generator and deriver
    expr_generator = RandomExpressionGenerator(
        num_vars=config.NUM_VARS,
        max_depth=config.TREE_DEPTH,
        domain_bounds=config.DOMAIN_BOUNDS,
        seed=config.SEED
    )
    
    deriver = Deriver(expr_generator.vars)
    
    dataset = []
    stats = {
        'total_attempted': 0,
        'total_success': 0,
        'total_failed': 0,
        'by_pde_type': {pde: 0 for pde in config.PDE_TYPES},
        'by_bc_type': {bc: 0 for bc in config.BC_TYPES}
    }
    
    # Calculate samples per combination
    samples_per_combination = config.SAMPLES_PER_PDE_TYPE // len(config.BC_TYPES)
    total_expected = len(config.PDE_TYPES) * len(config.BC_TYPES) * samples_per_combination
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PDE Dataset Generation")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - Tree depth: {config.TREE_DEPTH}")
        print(f"  - Number of variables: {config.NUM_VARS}")
        print(f"  - PDE types: {config.PDE_TYPES}")
        print(f"  - Boundary types: {config.BC_TYPES}")
        print(f"  - Samples per PDE type: {config.SAMPLES_PER_PDE_TYPE:,}")
        print(f"  - Samples per combination: {samples_per_combination:,}")
        print(f"  - Total samples: {total_expected:,}")
        print(f"{'='*60}\n")
    
    # Generate samples for each PDE type and BC type combination
    for pde_type in config.PDE_TYPES:
        for bc_type in config.BC_TYPES:
            
            if verbose:
                print(f"Generating {samples_per_combination:,} samples for {pde_type} + {bc_type}")
            
            # Progress bar
            iterator = range(samples_per_combination)
            if verbose:
                iterator = tqdm(iterator, desc=f"{pde_type[:4]}+{bc_type[:4]}")
            
            for idx, _ in enumerate(iterator):
                stats['total_attempted'] += 1
                
                sample = generate_single_sample(
                    expr_generator=expr_generator,
                    deriver=deriver,
                    pde_type=pde_type,
                    bc_type=bc_type,
                    config=config
                )
                
                if sample is not None:
                    dataset.append(sample)
                    stats['total_success'] += 1
                    stats['by_pde_type'][pde_type] += 1
                    stats['by_bc_type'][bc_type] += 1
                else:
                    stats['total_failed'] += 1
                
                # Periodic garbage collection and progress report
                if (idx + 1) % 100 == 0:
                    gc.collect()
                    
                    # Print intermediate progress every 500 samples
                    if verbose and (idx + 1) % 500 == 0:
                        tqdm.write(f"  Progress: {stats['total_success']:,} success, "
                                   f"{stats['total_failed']:,} failed")
    
    if verbose:
        print(f"\n{'='*60}")
        print("Generation Statistics:")
        print(f"  Total attempted: {stats['total_attempted']:,}")
        print(f"  Total successful: {stats['total_success']:,}")
        print(f"  Total failed: {stats['total_failed']:,}")
        if stats['total_attempted'] > 0:
            print(f"  Success rate: {stats['total_success']/stats['total_attempted']*100:.1f}%")
        print(f"\nBy PDE type:")
        for pde, count in stats['by_pde_type'].items():
            print(f"  {pde}: {count:,}")
        print(f"\nBy BC type:")
        for bc, count in stats['by_bc_type'].items():
            print(f"  {bc}: {count:,}")
        print(f"{'='*60}\n")
        
        # Print generator stats
        gen_stats = expr_generator.get_stats()
        print(f"Expression Generator Stats:")
        print(f"  Total generated: {gen_stats.get('total_generated', 0):,}")
        print(f"  Rejected (domain): {gen_stats.get('rejected_domain', 0):,}")
        print(f"  Rejected (complexity): {gen_stats.get('rejected_complexity', 0):,}")
        print(f"  Rejected (trivial): {gen_stats.get('rejected_trivial', 0):,}")
        print(f"  Rejected (timeout): {gen_stats.get('rejected_timeout', 0):,}")
    
    return dataset


def save_dataset(
    dataset: List[Dict[str, Any]],
    config: DataGenConfig,
    verbose: bool = True
) -> str:
    """
    Save the dataset to JSON file.
    """
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    filename = f"inference_test_data.json"
    output_path = os.path.join(config.OUTPUT_DIR, filename)
    
    # Save dataset
    if verbose:
        print(f"Saving {len(dataset):,} samples to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    if verbose:
        file_size = os.path.getsize(output_path)
        print(f"Saved! File size: {file_size / 1024 / 1024:.2f} MB")
    
    return output_path


# ==========================================================
# MAIN
# ==========================================================

def main():
    """Main entry point for data generation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PDE dataset for LLM training')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per PDE type (default: 1000)')
    parser.add_argument('--depth', type=int, default=3,
                       help='Tree depth for expression generation (default: 3)')
    parser.add_argument('--vars', type=int, default=2,
                       help='Number of variables (default: 2)')
    parser.add_argument('--output', type=str, default='./data',
                       help='Output directory (default: ./data)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--full', action='store_true',
                       help='Generate full dataset (198K samples) as in paper')
    
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.full:
        samples_per_pde = 99000
    else:
        samples_per_pde = args.samples
    
    config = DataGenConfig(
        TREE_DEPTH=args.depth,
        NUM_VARS=args.vars,
        SAMPLES_PER_PDE_TYPE=samples_per_pde,
        OUTPUT_DIR=args.output,
        SEED=args.seed
    )
    
    print("\n" + "="*60)
    print("PDE Dataset Generation for LLM Training")
    print("Based on: 'From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs'")
    print("="*60 + "\n")
    
    dataset = generate_dataset(config)
    output_path = save_dataset(dataset, config)
    
    print(f"\nDataset generation complete!")
    print(f"Output: {output_path}")
    
    # Show sample output
    if len(dataset) > 0:
        print("\n" + "="*60)
        print("Sample Data Points:")
        print("="*60)
        for i, dp in enumerate(dataset[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Input: {dp.get('input', 'N/A')[:100]}...")
            print(f"  Target: {dp.get('target', 'N/A')[:50]}...")
    
    return dataset


if __name__ == "__main__":
    main()
