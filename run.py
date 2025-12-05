import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to Python path
sys.path.insert(0, current_dir)

print(f"Python path: {sys.path}")
print(f"Current directory: {current_dir}")

# Try to import from core
try:
    from core import solutions, check_soln

    print("Successfully imported from core package")
    USE_CORE = True
except ImportError as e:
    print(f"Could not import from core: {e}")
    print("Using fallback implementation")
    USE_CORE = False

    # Fallback implementation
    import numpy as np


    def solutions(matrix_a, matrix_b, show=True):
        """Simplified version for fallback"""
        A = np.array(matrix_a, dtype=float).copy()
        b = np.array(matrix_b, dtype=float).copy()
        n = len(b)

        steps = []
        steps.append({
            'step': 0,
            'desc': 'Initial matrix',
            'A': A.copy(),
            'b_mat': b.copy()
        })

        for k in range(n):
            # Pivoting
            if abs(A[k, k]) < 1e-10:
                for i in range(k + 1, n):
                    if abs(A[i, k]) > abs(A[k, k]):
                        A[[k, i]] = A[[i, k]]
                        b[[k, i]] = b[[i, k]]
                        break

            # Normalize
            pivot = A[k, k]
            if pivot != 0:
                A[k] /= pivot
                b[k] /= pivot

            # Eliminate
            for i in range(n):
                if i != k:
                    factor = A[i, k]
                    if factor != 0:
                        A[i] -= factor * A[k]
                        b[i] -= factor * b[k]

        return b, A, steps


    def check_soln(origA, origB, solution):
        """Simplified check_soln"""
        A = np.array(origA, dtype=float)
        b = np.array(origB, dtype=float)
        x = np.array(solution, dtype=float)

        calc_b = A @ x
        error = np.abs(b - calc_b)

        return {
            'original': b,
            'calculated': calc_b,
            'error': error,
            'max_error': np.max(error),
            'is_correct': np.all(error < 1e-10)
        }

# Import and run the GUI
from app import GaussJordanApp

if __name__ == "__main__":
    print("Starting Gauss-Jordan Solver...")
    app = GaussJordanApp()
    app.mainloop()