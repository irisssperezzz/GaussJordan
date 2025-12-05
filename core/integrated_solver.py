import numpy as np
from solution import solutions, print_augmented_matrix, check_soln


class GaussJordanSolver:
    def __init__(self):
        self.steps = []
        self.solution = None
        self.rref_matrix = None

    def solve(self, matrix_a, matrix_b, show_steps=False):
        """
        Solve the linear system Ax = b using Gauss-Jordan elimination

        Parameters:
        matrix_a: 2D list or numpy array of coefficients
        matrix_b: 1D list or numpy array of constants
        show_steps: If True, returns detailed steps

        Returns:
        solution: The solution vector x
        rref_matrix: The reduced row echelon form
        steps: List of transformation steps (if show_steps=True)
        """
        try:
            # Convert to numpy arrays
            A = np.array(matrix_a, dtype=float)
            b = np.array(matrix_b, dtype=float)

            # Check if matrix is square
            if A.shape[0] != A.shape[1]:
                raise ValueError("Matrix A must be square")

            # Check dimensions match
            if A.shape[0] != len(b):
                raise ValueError("Matrix A and vector b dimensions don't match")

            # Use the existing solution function
            solution, rref_matrix, steps = solutions(matrix_a, matrix_b, show=show_steps)

            self.solution = solution
            self.rref_matrix = rref_matrix
            self.steps = steps

            return solution, rref_matrix, steps

        except Exception as e:
            raise Exception(f"Failed to solve: {str(e)}")

    def verify_solution(self, matrix_a, matrix_b, solution=None):
        """
        Verify the solution by checking if A*x ≈ b
        """
        if solution is None:
            solution = self.solution

        if solution is None:
            raise ValueError("No solution to verify")

        return check_soln(matrix_a, matrix_b, solution)

    def get_step_by_step(self):
        """
        Get formatted step-by-step solution
        """
        if not self.steps:
            raise ValueError("No steps available. Run solve() with show_steps=True first")

        formatted_steps = []
        for step in self.steps:
            formatted_step = {
                'step_number': step['step'],
                'description': step['desc'],
                'matrix': step['A'].tolist(),
                'vector': step['b_mat'].tolist()
            }
            formatted_steps.append(formatted_step)

        return formatted_steps