"""
Interactive Electromagnetic Fields Learning Environment

A Python package for learning and visualizing electromagnetic field concepts
through interactive Jupyter notebooks.
"""

__version__ = "0.1.0"
__author__ = "EM Fields Learning Team"

# Import main modules for easier access
try:
    from . import calculations
    from . import visualizations
except ImportError:
    # Allow package to be imported even if dependencies aren't installed yet
    pass
