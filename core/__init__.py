import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import your modules
try:
    from .solution import solutions, print_augmented_matrix, check_soln
    from .core import gaussjordan
    from .user_input import square_matrix, vector

    __all__ = [
        'solutions',
        'print_augmented_matrix',
        'check_soln',
        'gaussjordan',
        'square_matrix',
        'vector'
    ]

except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")