import customtkinter as ctk
import numpy as np
from tkinter import messagebox
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import from core
try:
    from core import solutions, check_soln

    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import from core. Error: {e}")
    print("Using fallback functions...")
    IMPORT_SUCCESS = False


    # Define fallback functions
    def solutions(matrix_a, matrix_b, show=True):
        # Simplified version for fallback
        import numpy as np

        A = np.array(matrix_a, dtype=float)
        b = np.array(matrix_b, dtype=float)
        n = len(b)

        # Create augmented matrix
        augmented = np.hstack([A, b.reshape(-1, 1)])

        steps = []
        steps.append({
            'step': 0,
            'desc': 'Initial matrix',
            'A': A.copy(),
            'b_mat': b.copy()
        })

        for i in range(n):
            # Pivot
            if abs(augmented[i, i]) < 1e-12:
                for j in range(i + 1, n):
                    if abs(augmented[j, i]) > abs(augmented[i, i]):
                        augmented[[i, j]] = augmented[[j, i]]
                        break

            # Normalize row
            pivot = augmented[i, i]
            if pivot != 0:
                augmented[i] = augmented[i] / pivot

            # Eliminate other rows
            for j in range(n):
                if j != i:
                    factor = augmented[j, i]
                    augmented[j] = augmented[j] - factor * augmented[i]

        solution = augmented[:, -1]
        A_final = augmented[:, :-1]

        return solution, A_final, steps


    def check_soln(origA, origB, solution):
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

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class GaussJordanSolverApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Gauss-Jordan Elimination Solver")
        self.geometry("1100x750")

        # Configure grid weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Variables
        self.matrix_size = ctk.IntVar(value=3)
        self.matrix_a = []
        self.matrix_b = []
        self.steps_data = []
        self.current_step = 0

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # --- TOP CONTROL FRAME ---
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        control_frame.grid_columnconfigure(1, weight=1)

        # Matrix size input
        ctk.CTkLabel(control_frame, text="Matrix Size (n x n):").grid(row=0, column=0, padx=10, pady=10)
        self.size_entry = ctk.CTkEntry(control_frame, width=60, textvariable=self.matrix_size)
        self.size_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Create Matrix Button
        self.create_btn = ctk.CTkButton(control_frame, text="Create Matrix",
                                        command=self.create_matrix, fg_color="green")
        self.create_btn.grid(row=0, column=2, padx=10, pady=10)

        # Solve Button
        self.solve_btn = ctk.CTkButton(control_frame, text="Solve",
                                       command=self.solve_matrix, fg_color="blue",
                                       state="disabled")
        self.solve_btn.grid(row=0, column=3, padx=10, pady=10)

        # Step-by-Step Button
        self.steps_btn = ctk.CTkButton(control_frame, text="Step-by-Step",
                                       command=self.show_steps, fg_color="orange",
                                       state="disabled")
        self.steps_btn.grid(row=0, column=4, padx=10, pady=10)

        # Clear Button
        self.clear_btn = ctk.CTkButton(control_frame, text="Clear All",
                                       command=self.clear_all, fg_color="red")
        self.clear_btn.grid(row=0, column=5, padx=10, pady=10)

        # Status label
        self.status_label = ctk.CTkLabel(control_frame, text="Ready", text_color="gray")
        self.status_label.grid(row=0, column=6, padx=20, pady=10)

        # --- MAIN CONTENT FRAME (for matrices) ---
        self.main_content = ctk.CTkFrame(self)
        self.main_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(0, weight=1)

        # Initially show empty state
        self.show_empty_state()

        # --- SOLUTION FRAME (hidden initially) ---
        self.solution_frame = ctk.CTkFrame(self)
        self.solution_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.solution_frame.grid_columnconfigure(0, weight=1)
        self.solution_frame.grid_forget()

        # --- STEPS FRAME (hidden initially) ---
        self.steps_frame = ctk.CTkFrame(self)
        self.steps_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.steps_frame.grid_columnconfigure(0, weight=1)
        self.steps_frame.grid_rowconfigure(1, weight=1)
        self.steps_frame.grid_forget()

        # Steps navigation
        steps_nav_frame = ctk.CTkFrame(self.steps_frame)
        steps_nav_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.prev_btn = ctk.CTkButton(steps_nav_frame, text="← Previous",
                                      command=self.prev_step, width=100)
        self.prev_btn.grid(row=0, column=0, padx=5, pady=5)

        self.step_label = ctk.CTkLabel(steps_nav_frame, text="Step 0/0")
        self.step_label.grid(row=0, column=1, padx=20, pady=5)

        self.next_btn = ctk.CTkButton(steps_nav_frame, text="Next →",
                                      command=self.next_step, width=100)
        self.next_btn.grid(row=0, column=2, padx=5, pady=5)

        self.close_steps_btn = ctk.CTkButton(steps_nav_frame, text="Close",
                                             command=self.close_steps, fg_color="red")
        self.close_steps_btn.grid(row=0, column=3, padx=5, pady=5)

        # Steps display area
        self.steps_display = ctk.CTkTextbox(self.steps_frame, font=("Consolas", 12))
        self.steps_display.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    def show_empty_state(self):
        # Clear main content
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Show empty state message
        label = ctk.CTkLabel(self.main_content,
                             text="Enter matrix size (2-10) and click 'Create Matrix' to begin",
                             font=("Arial", 16))
        label.pack(expand=True, padx=20, pady=20)

        # Show example
        example_frame = ctk.CTkFrame(self.main_content)
        example_frame.pack(pady=20)

        ctk.CTkLabel(example_frame, text="Example 3x3 system:", font=("Arial", 12, "bold")).pack(pady=5)
        ctk.CTkLabel(example_frame, text="2x + y - z = 8").pack()
        ctk.CTkLabel(example_frame, text="-3x - y + 2z = -11").pack()
        ctk.CTkLabel(example_frame, text="-2x + y + 2z = -3").pack()
        ctk.CTkLabel(example_frame, text="Solution: x=2, y=3, z=-1").pack(pady=5)

    def create_matrix(self):
        try:
            size = self.matrix_size.get()
            if size < 2 or size > 10:
                messagebox.showerror("Error", "Matrix size must be between 2 and 10")
                return

            # Clear main content
            for widget in self.main_content.winfo_children():
                widget.destroy()

            # Create matrix input frames
            matrix_frame = ctk.CTkFrame(self.main_content)
            matrix_frame.pack(fill="both", expand=True, padx=20, pady=20)
            matrix_frame.grid_columnconfigure(0, weight=1)
            matrix_frame.grid_columnconfigure(2, weight=1)

            # Matrix A Frame
            a_frame = ctk.CTkFrame(matrix_frame)
            a_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            ctk.CTkLabel(a_frame, text=f"Matrix A ({size}×{size})",
                         font=("Arial", 14, "bold")).pack(pady=10)

            # Create grid for matrix A
            self.a_entries = []
            for i in range(size):
                row_entries = []
                row_frame = ctk.CTkFrame(a_frame)
                row_frame.pack(pady=2)

                for j in range(size):
                    entry = ctk.CTkEntry(row_frame, width=70,
                                         placeholder_text=f"0")
                    if i == j:
                        entry.insert(0, "1")  # Default diagonal to 1
                    entry.pack(side="left", padx=2)
                    row_entries.append(entry)
                self.a_entries.append(row_entries)

            # Separator with equals sign
            eq_frame = ctk.CTkFrame(matrix_frame)
            eq_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")
            ctk.CTkLabel(eq_frame, text="×", font=("Arial", 20)).pack(pady=20)
            ctk.CTkLabel(eq_frame, text="x", font=("Arial", 20)).pack(pady=20)
            ctk.CTkLabel(eq_frame, text="=", font=("Arial", 20)).pack(pady=20)

            # Vector x (just labels)
            x_frame = ctk.CTkFrame(matrix_frame)
            x_frame.grid(row=0, column=2, padx=10, pady=10, sticky="ns")
            ctk.CTkLabel(x_frame, text="Vector x",
                         font=("Arial", 14, "bold")).pack(pady=10)

            for i in range(size):
                label = ctk.CTkLabel(x_frame, text=f"x{i + 1}",
                                     font=("Arial", 12), width=60, height=30)
                label.pack(pady=2)

            # Matrix B Frame (below)
            b_frame = ctk.CTkFrame(matrix_frame)
            b_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=20, sticky="ew")

            ctk.CTkLabel(b_frame, text=f"Vector b ({size}×1)",
                         font=("Arial", 14, "bold")).pack(pady=10)

            # Create entries for vector b
            self.b_entries = []
            b_row_frame = ctk.CTkFrame(b_frame)
            b_row_frame.pack()

            for i in range(size):
                entry = ctk.CTkEntry(b_row_frame, width=70,
                                     placeholder_text=f"b{i + 1}")
                entry.pack(side="left", padx=5)
                self.b_entries.append(entry)

            # Fill with example values if size=3
            if size == 3:
                example_A = [[2, 1, -1],
                             [-3, -1, 2],
                             [-2, 1, 2]]
                example_b = [8, -11, -3]

                for i in range(size):
                    for j in range(size):
                        self.a_entries[i][j].delete(0, "end")
                        self.a_entries[i][j].insert(0, str(example_A[i][j]))

                for i in range(size):
                    self.b_entries[i].delete(0, "end")
                    self.b_entries[i].insert(0, str(example_b[i]))

            # Enable solve button
            self.solve_btn.configure(state="normal")
            self.status_label.configure(text=f"Created {size}x{size} matrix", text_color="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create matrix: {str(e)}")

    def get_matrix_values(self):
        try:
            size = self.matrix_size.get()
            matrix_a = []
            matrix_b = []

            # Get matrix A values
            for i in range(size):
                row = []
                for j in range(size):
                    val = self.a_entries[i][j].get()
                    if val == "":
                        val = "0"
                    row.append(float(val))
                matrix_a.append(row)

            # Get vector b values
            for i in range(size):
                val = self.b_entries[i].get()
                if val == "":
                    val = "0"
                matrix_b.append(float(val))

            return matrix_a, matrix_b

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
            return None, None

    def solve_matrix(self):
        matrix_a, matrix_b = self.get_matrix_values()
        if matrix_a is None:
            return

        try:
            self.status_label.configure(text="Solving...", text_color="orange")
            self.update()

            # Solve using the solution module
            if IMPORT_SUCCESS:
                solution, _, steps = solutions(matrix_a, matrix_b, show=False)
            else:
                solution, _, steps = solutions(matrix_a, matrix_b, show=False)

            # Store steps for step-by-step view
            self.steps_data = steps
            self.current_step = 0

            # Show solution frame
            self.show_solution(solution, matrix_a, matrix_b)

            # Enable step-by-step button
            self.steps_btn.configure(state="normal")
            self.status_label.configure(text="Solved successfully!", text_color="green")

        except Exception as e:
            self.status_label.configure(text="Error solving", text_color="red")
            messagebox.showerror("Error", f"Failed to solve: {str(e)}")

    def show_solution(self, solution, matrix_a, matrix_b):
        # Hide main content
        self.main_content.grid_forget()

        # Show solution frame
        self.solution_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Clear previous solution
        for widget in self.solution_frame.winfo_children():
            widget.destroy()

        # Solution header
        header_frame = ctk.CTkFrame(self.solution_frame)
        header_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(header_frame, text="Solution",
                     font=("Arial", 20, "bold")).pack(side="left", padx=10)

        back_btn = ctk.CTkButton(header_frame, text="← Back to Matrix",
                                 command=self.back_to_matrix, fg_color="gray", width=150)
        back_btn.pack(side="right", padx=10)

        # Display solution vector
        sol_frame = ctk.CTkFrame(self.solution_frame)
        sol_frame.pack(fill="both", expand=True, padx=40, pady=20)

        ctk.CTkLabel(sol_frame, text="Solution Vector x:",
                     font=("Arial", 16)).pack(pady=10)

        # Create a grid for solution values
        size = len(solution)
        for i in range(size):
            row_frame = ctk.CTkFrame(sol_frame)
            row_frame.pack(pady=5)

            ctk.CTkLabel(row_frame, text=f"x{i + 1} = ",
                         font=("Arial", 14)).pack(side="left", padx=5)
            ctk.CTkLabel(row_frame, text=f"{solution[i]:.8f}",
                         font=("Courier", 14, "bold"),
                         text_color="green").pack(side="left", padx=5)

        # Buttons frame
        btn_frame = ctk.CTkFrame(self.solution_frame)
        btn_frame.pack(pady=20)

        # Verify solution button
        verify_btn = ctk.CTkButton(btn_frame, text="Verify Solution",
                                   command=lambda: self.verify_solution(matrix_a, matrix_b, solution),
                                   fg_color="blue", width=150)
        verify_btn.pack(side="left", padx=10)

        # Step-by-step button
        steps_btn = ctk.CTkButton(btn_frame, text="Show Step-by-Step",
                                  command=self.show_steps,
                                  fg_color="orange", width=150)
        steps_btn.pack(side="left", padx=10)

        # New problem button
        new_btn = ctk.CTkButton(btn_frame, text="New Problem",
                                command=self.clear_all,
                                fg_color="green", width=150)
        new_btn.pack(side="left", padx=10)

    def verify_solution(self, matrix_a, matrix_b, solution):
        try:
            result = check_soln(matrix_a, matrix_b, solution)

            # Show verification results
            verif_window = ctk.CTkToplevel(self)
            verif_window.title("Solution Verification")
            verif_window.geometry("500x400")

            ctk.CTkLabel(verif_window, text="Verification Results",
                         font=("Arial", 16, "bold")).pack(pady=15)

            # Create a scrollable frame for results
            scroll_frame = ctk.CTkScrollableFrame(verif_window, height=250)
            scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

            # Original b
            ctk.CTkLabel(scroll_frame, text="Original vector b:",
                         font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
            orig_text = "  ".join(f"{val:10.6f}" for val in result['original'])
            ctk.CTkLabel(scroll_frame, text=orig_text, font=("Courier", 11)).pack(anchor="w", pady=2)

            # Calculated A*x
            ctk.CTkLabel(scroll_frame, text="Calculated A*x:",
                         font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
            calc_text = "  ".join(f"{val:10.6f}" for val in result['calculated'])
            ctk.CTkLabel(scroll_frame, text=calc_text, font=("Courier", 11)).pack(anchor="w", pady=2)

            # Error
            ctk.CTkLabel(scroll_frame, text="Error (|b - A*x|):",
                         font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
            error_text = "  ".join(f"{val:10.6e}" for val in result['error'])
            ctk.CTkLabel(scroll_frame, text=error_text, font=("Courier", 11)).pack(anchor="w", pady=2)

            # Maximum error
            ctk.CTkLabel(scroll_frame, text=f"Maximum error: {result['max_error']:.10e}",
                         font=("Arial", 12, "bold")).pack(anchor="w", pady=10)

            # Conclusion
            if result['is_correct']:
                ctk.CTkLabel(scroll_frame, text="✓ Solution is correct!",
                             text_color="green", font=("Arial", 12, "bold")).pack(pady=10)
            else:
                ctk.CTkLabel(scroll_frame, text="✗ Solution may have errors",
                             text_color="red", font=("Arial", 12, "bold")).pack(pady=10)

            # Close button
            ctk.CTkButton(verif_window, text="Close",
                          command=verif_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

    def back_to_matrix(self):
        # Hide solution frame
        self.solution_frame.grid_forget()

        # Show main content
        self.main_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

    def show_steps(self):
        if not self.steps_data:
            messagebox.showinfo("No Steps", "No step-by-step data available")
            return

        # Hide main content and solution frame
        self.main_content.grid_forget()
        self.solution_frame.grid_forget()

        # Show steps frame
        self.steps_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Display first step
        self.current_step = 0
        self.display_current_step()

    def display_current_step(self):
        if not self.steps_data or self.current_step >= len(self.steps_data):
            return

        step = self.steps_data[self.current_step]

        # Clear display
        self.steps_display.delete("1.0", "end")

        # Display step information
        self.steps_display.insert("1.0", f"Step {step['step']}: {step['desc']}\n")
        self.steps_display.insert("end", "=" * 60 + "\n\n")

        # Display augmented matrix
        try:
            A = step['A']
            b = step['b_mat']

            # Format matrix for display
            n = len(b)
            for i in range(n):
                row_str = "["
                for j in range(n):
                    row_str += f"{A[i, j]:10.6f}"
                row_str += f" | {b[i]:10.6f} ]\n"
                self.steps_display.insert("end", row_str)
        except:
            # If there's an error displaying, show raw data
            self.steps_display.insert("end", f"Matrix A: {step.get('A', 'N/A')}\n")
            self.steps_display.insert("end", f"Vector b: {step.get('b_mat', 'N/A')}\n")

        # Update step label
        self.step_label.configure(text=f"Step {self.current_step + 1}/{len(self.steps_data)}")

        # Update button states
        self.prev_btn.configure(state="normal" if self.current_step > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_step < len(self.steps_data) - 1 else "disabled")

    def next_step(self):
        if self.current_step < len(self.steps_data) - 1:
            self.current_step += 1
            self.display_current_step()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.display_current_step()

    def close_steps(self):
        # Hide steps frame
        self.steps_frame.grid_forget()

        # Show main content
        self.main_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

    def clear_all(self):
        # Clear all data
        self.matrix_size.set(3)
        self.steps_data = []
        self.current_step = 0

        # Reset buttons
        self.solve_btn.configure(state="disabled")
        self.steps_btn.configure(state="disabled")

        # Hide frames
        self.solution_frame.grid_forget()
        self.steps_frame.grid_forget()

        # Show empty state
        self.show_empty_state()
        self.status_label.configure(text="Ready", text_color="gray")


if __name__ == "__main__":
    app = GaussJordanSolverApp()
    app.mainloop()