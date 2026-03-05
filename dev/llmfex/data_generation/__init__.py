"""
PDE Data Generation Package

This package generates training data for fine-tuning LLMs to predict 
symbolic operators in PDE solutions, following the methodology from:

"From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs"
by Bhatnagar, Liang, Patel, and Yang

Usage:
    python DGen.py --samples 1000 --output ./data

For full dataset (198K samples):
    python DGen.py --depth 3 --vars 3 --full --output ./data/ --seed 42
"""

from .PDE import PDE
from .REGen import RandomExpressionGenerator, OperatorConfig
from .Deriver import Deriver, BoundaryFace, create_boundary_face
from .RPNC import RPNConverter, assemble_data_point

__version__ = '1.0.0'
__all__ = [
    'PDE',
    'RandomExpressionGenerator', 
    'OperatorConfig',
    'Deriver',
    'BoundaryFace',
    'create_boundary_face',
    'RPNConverter',
    'assemble_data_point'
]
