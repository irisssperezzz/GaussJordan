import customtkinter as ctk
import numpy as np
from tkinter import messagebox

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


'''
One application based on the console-based program of Iris
and Initial design of Jovany for the GUI
'''
class GaussJordanSolver:

    @staticmethod
    def solve(matrix_a, matrix_b, show_steps=True):
        """
        Solving Matrix Ax = Matrix b (vector) using Gauss-Jordan elimination with step tracking
        """
        try:
            A = np.array(matrix_a, dtype=float).copy()
            b = np.array(matrix_b, dtype=float).copy()
            n = len(b)

            steps = []
            step_num = 0

            # Initial state
            steps.append({
                'step': step_num,
                'desc': 'Initial augmented matrix',
                'A': A.copy(),
                'b': b.copy(),
                'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
            })
            step_num += 1

            for k in range(n):
                # Step 1: Partial pivoting if needed
                if abs(A[k, k]) < 1e-10:
                    for i in range(k + 1, n):
                        if abs(A[i, k]) > abs(A[k, k]):
                            # Swap rows
                            A[[k, i]] = A[[i, k]]
                            b[[k, i]] = b[[i, k]]

                            steps.append({
                                'step': step_num,
                                'desc': f'Pivot: Swap row {k + 1} with row {i + 1}',
                                'A': A.copy(),
                                'b': b.copy(),
                                'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
                            })
                            step_num += 1
                            break

                # Step 2: Normalize pivot row
                pivot = A[k, k]
                if abs(pivot) > 1e-12:
                    A[k] = A[k] / pivot
                    b[k] = b[k] / pivot

                    steps.append({
                        'step': step_num,
                        'desc': f'Normalize row {k + 1}: Divide by {pivot:.4f}',
                        'A': A.copy(),
                        'b': b.copy(),
                        'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
                    })
                    step_num += 1

                # Step 3: Eliminate from other rows
                for i in range(n):
                    if i != k and abs(A[i, k]) > 1e-12:
                        factor = A[i, k]
                        A[i] = A[i] - factor * A[k]
                        b[i] = b[i] - factor * b[k]

                        steps.append({
                            'step': step_num,
                            'desc': f'Eliminate: R{i + 1} ← R{i + 1} - ({factor:.4f})×R{k + 1}',
                            'A': A.copy(),
                            'b': b.copy(),
                            'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
                        })
                        step_num += 1

            # Final step
            steps.append({
                'step': step_num,
                'desc': 'Final solution (Reduced Row Echelon Form)',
                'A': A.copy(),
                'b': b.copy(),
                'matrix': np.hstack([A.copy(), b.copy().reshape(-1, 1)])
            })

            return b, A, steps

        except Exception as e:
            raise Exception(f"Solution failed: {str(e)}")

    @staticmethod
    def verify(matrix_a, matrix_b, solution):
        """Verify the solution by checking A*x ≈ b"""
        A = np.array(matrix_a, dtype=float)
        b = np.array(matrix_b, dtype=float)
        x = np.array(solution, dtype=float)

        calculated = A @ x
        error = np.abs(b - calculated)

        return {
            'original': b,
            'calculated': calculated,
            'error': error,
            'max_error': np.max(error),
            'is_correct': np.all(error < 1e-10)
        }


class GaussJordanApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Gauss-Jordan Calculator")
        self.geometry("1000x800")

        # grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # var
        self.matrix_size = ctk.IntVar(value=3)
        self.steps_data = []
        self.current_step = 0
        self.solution = None

        # ui creation
        self.create_widgets()

        # example data
        self.after(100, self.set_default_example)

    def create_widgets(self):
        # --- TOP CONTROL FRAME ---
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        # matrix size
        ctk.CTkLabel(control_frame, text="Matrix Size (n×n):").pack(side="left", padx=(10, 5))
        self.size_entry = ctk.CTkEntry(control_frame, width=60, textvariable=self.matrix_size)
        self.size_entry.pack(side="left", padx=5)

        # matrix button
        self.create_btn = ctk.CTkButton(control_frame, text="Create Matrix",
                                        command=self.create_matrix, fg_color="green")
        self.create_btn.pack(side="left", padx=10)

        # solve button
        self.solve_btn = ctk.CTkButton(control_frame, text="Solve",
                                       command=self.solve_system, fg_color="blue",
                                       state="disabled")
        self.solve_btn.pack(side="left", padx=10)

        # clear button
        self.clear_btn = ctk.CTkButton(control_frame, text="Clear All",
                                       command=self.clear_all, fg_color="red")
        self.clear_btn.pack(side="left", padx=10)

        # status label
        # purpose to tell whether the matrix is solvable or not
        self.status_label = ctk.CTkLabel(control_frame, text="Ready", text_color="gray")
        self.status_label.pack(side="left", padx=20)

        # --- MAIN CONTENT AREA ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # welcome screen
        self.show_welcome_screen()

    def show_welcome_screen(self):
        """Display welcome/instructions screen"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        welcome_frame = ctk.CTkFrame(self.main_frame)
        welcome_frame.pack(fill="both", expand=True, padx=50, pady=50)

        # Title
        ctk.CTkLabel(welcome_frame, text="Gauss-Jordan Elimination Solver",
                     font=("Consolas", 24, "bold")).pack(pady=30)

        # Instructions
        instructions = [
            "1. Enter the square matrix size (example.g., 3 for 3×3)",
            "2. Click 'Create Matrix' to generate input fields",
            "3. Enter values for the coefficient matrix and constant matrix",
            "4. Click 'Solve' to compute solution",
            "5. View step-by-step solution"
        ]

        for instruction in instructions:
            ctk.CTkLabel(welcome_frame, text=instruction,
                         font=("Consolas", 14)).pack(pady=5)

        # Example frame
        example_frame = ctk.CTkFrame(welcome_frame)
        example_frame.pack(pady=30, fill="x", padx=50)

        ctk.CTkLabel(example_frame, text="Example 3×3 System:",
                     font=("Consolas", 16, "bold")).pack(pady=10)

        example = """2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3

Solution: x = 2, y = 3, z = -1"""

        ctk.CTkLabel(example_frame, text=example,
                     font=("Courier", 12), justify="left").pack(pady=10)

        # Quick start button
        ctk.CTkButton(welcome_frame, text="Load Example",
                      command=self.load_example, fg_color="orange").pack(pady=20)

    def set_default_example(self):
        # set values
        self.matrix_size.set(3)
        self.create_matrix()

    def load_example(self):
        # load values
        self.matrix_size.set(3)
        self.create_matrix()

    # input fields for the Matrix Ax and Vector b
    def create_matrix(self):
        try:
            size = self.matrix_size.get()
            if size < 2 or size > 8:
                messagebox.showerror("Error", "Matrix size must be between 2 and 8")
                return

            # Cmain frame
            for widget in self.main_frame.winfo_children():
                widget.destroy()

            # input area
            input_frame = ctk.CTkFrame(self.main_frame)
            input_frame.pack(fill="both", expand=True, padx=20, pady=20)

            # Title
            ctk.CTkLabel(input_frame, text=f"Enter {size}×{size} System",
                         font=("Consolas", 18, "bold")).pack(pady=10)

            # matrix input area
            matrix_area = ctk.CTkFrame(input_frame)
            matrix_area.pack(pady=20)

            # matrix A frame
            a_frame = ctk.CTkFrame(matrix_area)
            a_frame.grid(row=0, column=0, padx=20, pady=10)

            ctk.CTkLabel(a_frame, text="Matrix A",
                         font=("Consolas", 14, "bold")).pack(pady=5)

            # create A entries grid
            self.a_entries = []
            for i in range(size):
                row_frame = ctk.CTkFrame(a_frame)
                row_frame.pack(pady=2)
                row_entries = []

                for j in range(size):
                    entry = ctk.CTkEntry(row_frame, width=70,
                                         placeholder_text="0")
                    # set default values for example
                    if size == 3 and i == j:
                        if i == 0:
                            entry.insert(0, "2")
                        elif i == 1:
                            entry.insert(0, "-3")
                        else:
                            entry.insert(0, "-2")
                    elif size == 3 and i == 0 and j == 1:
                        entry.insert(0, "1")
                    elif size == 3 and i == 0 and j == 2:
                        entry.insert(0, "-1")
                    elif size == 3 and i == 1 and j == 0:
                        entry.insert(0, "-1")
                    elif size == 3 and i == 1 and j == 2:
                        entry.insert(0, "2")
                    elif size == 3 and i == 2 and j == 0:
                        entry.insert(0, "1")
                    elif size == 3 and i == 2 and j == 2:
                        entry.insert(0, "2")

                    entry.pack(side="left", padx=2)
                    row_entries.append(entry)
                self.a_entries.append(row_entries)

            # multiplication and equals symbols
            sym_frame = ctk.CTkFrame(matrix_area)
            sym_frame.grid(row=0, column=1, padx=10, pady=10)

            ctk.CTkLabel(sym_frame, text="×", font=("Consolas", 24)).pack(pady=15)
            ctk.CTkLabel(sym_frame, text="X", font=("Consolas", 24)).pack(pady=15)
            ctk.CTkLabel(sym_frame, text="=", font=("Consolas", 24)).pack(pady=15)

            # vector X labels
            x_frame = ctk.CTkFrame(matrix_area)
            x_frame.grid(row=0, column=2, padx=10, pady=10)

            ctk.CTkLabel(x_frame, text="Vector X",
                         font=("Consolas", 14, "bold")).pack(pady=5)

            for i in range(size):
                ctk.CTkLabel(x_frame, text=f"x{i + 1}",
                             font=("Consolas", 14), width=70, height=30).pack(pady=2)

            # vector b frame
            b_frame = ctk.CTkFrame(matrix_area)
            b_frame.grid(row=0, column=3, padx=20, pady=10)

            ctk.CTkLabel(b_frame, text="Vector b",
                         font=("Consolas", 14, "bold")).pack(pady=5)

            # create b entries
            self.b_entries = []
            for i in range(size):
                entry = ctk.CTkEntry(b_frame, width=70,
                                     placeholder_text="0")
                # Set default values for example
                if size == 3:
                    if i == 0:
                        entry.insert(0, "8")
                    elif i == 1:
                        entry.insert(0, "-11")
                    else:
                        entry.insert(0, "-3")

                entry.pack(pady=2)
                self.b_entries.append(entry)

            # button frame
            btn_frame = ctk.CTkFrame(input_frame)
            btn_frame.pack(pady=20)

            # solve button
            self.solve_btn.configure(state="normal")
            ctk.CTkButton(btn_frame, text="Solve System",
                          command=self.solve_system,
                          fg_color="green", width=150, height=40,
                          font=("Consolas", 14)).pack(side="left", padx=10)

            # Step-by-step button (initially disabled)
            self.steps_btn = ctk.CTkButton(btn_frame, text="Show Steps",
                                           command=self.show_steps_viewer,
                                           fg_color="orange", width=150, height=40,
                                           font=("Consolas", 14), state="disabled")
            self.steps_btn.pack(side="left", padx=10)

            # new system button
            ctk.CTkButton(btn_frame, text="New System",
                          command=self.show_welcome_screen,
                          fg_color="gray", width=150, height=40,
                          font=("Consolas", 14)).pack(side="left", padx=10)

            self.status_label.configure(text=f"Created {size}×{size} system", text_color="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create matrix: {str(e)}")

    # getting the values from inputs
    def get_matrix_values(self):
        try:
            size = self.matrix_size.get()
            matrix_a = []
            matrix_b = []

            # Get matrix A
            for i in range(size):
                row = []
                for j in range(size):
                    val = self.a_entries[i][j].get()
                    if val == "":
                        val = "0"
                    row.append(float(val))
                matrix_a.append(row)

            # Get vector b
            for i in range(size):
                val = self.b_entries[i].get()
                if val == "":
                    val = "0"
                matrix_b.append(float(val))

            return matrix_a, matrix_b

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
            return None, None

    # solve through linear system
    def solve_system(self):
        matrix_a, matrix_b = self.get_matrix_values()
        if matrix_a is None:
            return

        try:
            self.status_label.configure(text="Solving...", text_color="orange")
            self.update()

            # create solver instance
            solver = GaussJordanSolver()

            # solve the system
            solution, rref, steps = solver.solve(matrix_a, matrix_b)

            # store for later use
            self.solution = solution
            self.steps_data = steps
            self.matrix_a = matrix_a
            self.matrix_b = matrix_b
            self.current_step = 0

            # enable steps button
            self.steps_btn.configure(state="normal")

            # show solution
            self.show_solution(solution, matrix_a, matrix_b)

            self.status_label.configure(text="Solved successfully!", text_color="green")

        except Exception as e:
            self.status_label.configure(text="Error", text_color="red")
            messagebox.showerror("Error", f"Failed to solve system: {str(e)}")

    # display solution
    def show_solution(self, solution, matrix_a, matrix_b):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        solution_frame = ctk.CTkFrame(self.main_frame)
        solution_frame.pack(fill="both", expand=True, padx=30, pady=30)

        # header
        header_frame = ctk.CTkFrame(solution_frame)
        header_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(header_frame, text="Solution",
                     font=("Consolas", 24, "bold")).pack(side="left", padx=10)

        ctk.CTkButton(header_frame, text="← Back",
                      command=self.create_matrix,
                      fg_color="gray", width=100).pack(side="right", padx=10)

        # solution values
        ctk.CTkLabel(solution_frame, text="Solution Vector X:",
                     font=("Consolas", 18)).pack(pady=20)

        size = len(solution)
        for i in range(size):
            row_frame = ctk.CTkFrame(solution_frame)
            row_frame.pack(pady=5)

            ctk.CTkLabel(row_frame, text=f"x{i + 1} = ",
                         font=("Consolas", 16)).pack(side="left", padx=5)

            # highlight the value
            value_label = ctk.CTkLabel(row_frame, text=f"{solution[i]:.8f}",
                                       font=("Courier", 16, "bold"),
                                       text_color="green")
            value_label.pack(side="left", padx=5)

        # verify button
        ctk.CTkButton(solution_frame, text="Verify Solution",
                      command=self.verify_solution,
                      fg_color="blue", width=200, height=40,
                      font=("Consolas", 14)).pack(pady=30)

        # steps button
        ctk.CTkButton(solution_frame, text="View Step-by-Step Solution",
                      command=self.show_steps_viewer,
                      fg_color="orange", width=200, height=40,
                      font=("Consolas", 14)).pack(pady=10)

    # verification
    def verify_solution(self):
        if self.solution is None:
            return

        try:
            solver = GaussJordanSolver()
            result = solver.verify(self.matrix_a, self.matrix_b, self.solution)

            # create verification window
            verify_window = ctk.CTkToplevel(self)
            verify_window.title("Solution Verification")
            verify_window.geometry("600x500")

            # Title
            ctk.CTkLabel(verify_window, text="Verification Results",
                         font=("Consolas", 20, "bold")).pack(pady=20)

            # create scrollable content
            scroll_frame = ctk.CTkScrollableFrame(verify_window)
            scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

            # original b
            ctk.CTkLabel(scroll_frame, text="Original b:",
                         font=("Consolas", 14, "bold")).pack(anchor="w", pady=5)

            orig_text = ""
            for i, val in enumerate(result['original']):
                orig_text += f"b{i + 1} = {val:12.8f}\n"
            ctk.CTkLabel(scroll_frame, text=orig_text,
                         font=("Courier", 12), justify="left").pack(anchor="w")

            # calculated A*x
            ctk.CTkLabel(scroll_frame, text="Calculated A*x:",
                         font=("Consolas", 14, "bold")).pack(anchor="w", pady=(20, 5))

            calc_text = ""
            for i, val in enumerate(result['calculated']):
                calc_text += f"b{i + 1} = {val:12.8f}\n"
            ctk.CTkLabel(scroll_frame, text=calc_text,
                         font=("Courier", 12), justify="left").pack(anchor="w")

            # Errors
            ctk.CTkLabel(scroll_frame, text="Errors (|b - A*x|):",
                         font=("Consolas", 14, "bold")).pack(anchor="w", pady=(20, 5))

            error_text = ""
            for i, val in enumerate(result['error']):
                error_text += f"Error {i + 1} = {val:12.2e}\n"
            ctk.CTkLabel(scroll_frame, text=error_text,
                         font=("Courier", 12), justify="left").pack(anchor="w")

            # max error
            ctk.CTkLabel(scroll_frame,
                         text=f"Maximum Error: {result['max_error']:.2e}",
                         font=("Consolas", 14, "bold")).pack(anchor="w", pady=20)

            # conclusion
            if result['is_correct']:
                ctk.CTkLabel(scroll_frame, text="✓ Solution is CORRECT",
                             text_color="green", font=("Consolas", 16, "bold")).pack(pady=10)
            else:
                ctk.CTkLabel(scroll_frame, text="✗ Solution has ERRORS",
                             text_color="red", font=("Consolas", 16, "bold")).pack(pady=10)

            # close button
            ctk.CTkButton(verify_window, text="Close",
                          command=verify_window.destroy,
                          fg_color="gray").pack(pady=20)

        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

    def show_steps_viewer(self):
        """Show step-by-step solution viewer"""
        if not self.steps_data:
            messagebox.showinfo("No Steps", "No step-by-step data available")
            return

        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # create steps viewer
        steps_frame = ctk.CTkFrame(self.main_frame)
        steps_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # header
        header_frame = ctk.CTkFrame(steps_frame)
        header_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(header_frame, text="Step-by-Step Solution",
                     font=("Consolas", 20, "bold")).pack(side="left", padx=10)

        ctk.CTkButton(header_frame, text="← Back to Solution",
                      command=lambda: self.show_solution(self.solution, self.matrix_a, self.matrix_b),
                      fg_color="gray").pack(side="right", padx=10)

        # navigation controls
        nav_frame = ctk.CTkFrame(steps_frame)
        nav_frame.pack(fill="x", pady=10)

        self.prev_btn = ctk.CTkButton(nav_frame, text="◀ Previous",
                                      command=self.prev_step,
                                      state="disabled", width=120)
        self.prev_btn.pack(side="left", padx=10)

        self.step_label = ctk.CTkLabel(nav_frame, text="Step 1/1",
                                       font=("Consolas", 14))
        self.step_label.pack(side="left", padx=20)

        self.next_btn = ctk.CTkButton(nav_frame, text="Next ▶",
                                      command=self.next_step,
                                      state="normal", width=120)
        self.next_btn.pack(side="left", padx=10)

        # steps display area
        display_frame = ctk.CTkFrame(steps_frame)
        display_frame.pack(fill="both", expand=True, pady=10)

        # text display for matrix
        self.matrix_display = ctk.CTkTextbox(display_frame, font=("Courier", 14))
        self.matrix_display.pack(fill="both", expand=True, padx=10, pady=10)

        # description
        self.desc_label = ctk.CTkLabel(steps_frame, text="",
                                       font=("Consolas", 14, "bold"))
        self.desc_label.pack(pady=10)

        # show first step
        self.current_step = 0
        self.show_current_step()

    # show current step
    def show_current_step(self):
        if not self.steps_data or self.current_step >= len(self.steps_data):
            return

        step = self.steps_data[self.current_step]

        # clear display
        self.matrix_display.delete("1.0", "end")

        # update description
        self.desc_label.configure(text=f"Step {step['step']}: {step['desc']}")

        # display matrix
        try:
            if 'matrix' in step:
                matrix = step['matrix']
                n = matrix.shape[0]

                display_text = ""
                for i in range(n):
                    row_text = "[ "
                    for j in range(n):
                        row_text += f"{matrix[i, j]:10.6f} "
                    row_text += "| "
                    row_text += f"{matrix[i, n]:10.6f} ]\n"
                    display_text += row_text

                self.matrix_display.insert("1.0", display_text)
            else:
                # Fallback: use A and b separately
                A = step['A']
                b = step['b']
                n = len(b)

                display_text = ""
                for i in range(n):
                    row_text = "[ "
                    for j in range(n):
                        row_text += f"{A[i, j]:10.6f} "
                    row_text += "| "
                    row_text += f"{b[i]:10.6f} ]\n"
                    display_text += row_text

                self.matrix_display.insert("1.0", display_text)

        except Exception as e:
            self.matrix_display.insert("1.0", f"Error displaying matrix: {str(e)}")

        # update navigation
        self.step_label.configure(text=f"Step {self.current_step + 1}/{len(self.steps_data)}")
        self.prev_btn.configure(state="normal" if self.current_step > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_step < len(self.steps_data) - 1 else "disabled")

    def next_step(self):
        """Go to next step"""
        if self.current_step < len(self.steps_data) - 1:
            self.current_step += 1
            self.show_current_step()

    def prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.show_current_step()

    def clear_all(self):
        """CLEAR screen, data, values"""
        self.matrix_size.set(3)
        self.steps_data = []
        self.current_step = 0
        self.solution = None
        self.solve_btn.configure(state="disabled")
        self.status_label.configure(text="Ready", text_color="gray")
        self.show_welcome_screen()


if __name__ == "__main__":
    app = GaussJordanApp()
    app.mainloop()
