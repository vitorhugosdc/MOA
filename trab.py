import numpy as np

def print_tableau(T, step):
    print(f"Tableau at step {step}:")
    print(T, "\n")

def pivot(T, pivot_row, pivot_col):
    # Divide all elements in the pivot row by the pivot element
    T[pivot_row, :] /= T[pivot_row, pivot_col]
    # Subtract multiples of the pivot row from all other rows
    for i in range(len(T)):
        if i != pivot_row:
            T[i, :] -= T[i, pivot_col] * T[pivot_row, :]
    return T

def simplex(c, A, b):
    num_vars = len(c)
    # Construct the initial simplex tableau
    T = np.hstack([A, np.eye(len(A)), b.reshape(-1, 1)])
    T = np.vstack([T, np.hstack([-c, np.zeros(len(A) + 1)])])
    print_tableau(T, "Initial")
    
    while True:
        # Check if we have reached the optimal solution
        if np.all(T[-1, :-1] >= 0):
            print("Optimal solution found.")
            break
        
        # Choose entering variable (most negative coefficient in the objective function)
        pivot_col = np.argmin(T[-1, :-1])
        
        # Ratio test to choose leaving variable
        ratios = np.divide(T[:-1, -1], T[:-1, pivot_col], out=np.full_like(T[:-1, -1], np.inf), where=T[:-1, pivot_col]>0)
        pivot_row = np.argmin(ratios)
        
        # If all entries in the pivot column are <= 0, the solution is unbounded
        if np.all(T[:-1, pivot_col] <= 0):
            print("Unbounded solution.")
            break
        
        # Pivot
        T = pivot(T, pivot_row, pivot_col)
        print_tableau(T, f"Pivot at row {pivot_row}, col {pivot_col}")
    
    return T

# Coefficients of the objective function
c = np.array([3, 5])

# Coefficients of the constraints
A = np.array([
    [3, 2],
    [1, 0],
    [0, 2],
])

# Right-hand side of the constraints
b = np.array([18, 4, 12])

# Run the simplex algorithm
final_tableau = simplex(c, A, b)

# Extract solution
if final_tableau is not None:
    solution = final_tableau[:-1, -1]
    objective_value = -final_tableau[-1, -1]
    print("Solution:", solution)
    print("Objective value:", objective_value)
