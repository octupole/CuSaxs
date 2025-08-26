import argparse
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

def read_file(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]

def interpolate_data(x, y, new_x):
    spline = interpolate.splrep(x, y)
    return interpolate.splev(new_x, spline)

def fit_function(alpha, q, I_exp, I_sol, q_min_fit, q_max_fit):
    mask = (q > q_min_fit) & (q <= q_max_fit)
    return np.sum((I_exp[mask] - alpha * I_sol[mask])**2)

def constraint(alpha, q, I_exp, I_sol, q_min_fit, q_max_fit):
    mask = (q > q_min_fit) & (q <= q_max_fit)
    return np.min(I_exp[mask] - alpha * I_sol[mask]) - 10.0

def plot_results(q, I_exp, I_sol, alpha, difference, q_min_fit, q_max_fit, q_max_diff):
    plt.figure(figsize=(12, 12))
    
    # Plot original and fitted functions with log y-axis
    plt.subplot(2, 1, 1)
    plt.semilogy(q, I_exp, label='I_exp')
    plt.semilogy(q, np.abs(alpha * I_sol), label=f'|α * I_sol| (α = {alpha:.4f})')
    plt.axvline(q_min_fit, color='r', linestyle='--', label='Fit range')
    plt.axvline(q_max_fit, color='r', linestyle='--')
    plt.legend()
    plt.title('Original and Fitted Functions (log scale)')
    plt.xlabel('q')
    plt.ylabel('log(I(q))')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot difference with log y-axis
    plt.subplot(2, 1, 2)
    plt.semilogy(q[q <= q_max_diff], difference[q <= q_max_diff])
    plt.axvline(q_max_diff, color='g', linestyle='--', label='Diff range')
    plt.axhline(10.0, color='r', linestyle='--', label='Constraint')
    plt.legend()
    plt.title(f'Difference: I_exp - α * I_sol (log scale, q ≤ {q_max_diff})')
    plt.xlabel('q')
    plt.ylabel('Difference')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def main(f_exp, f_sol, output, num_points=1000, q_min_fit=1.7, q_max_fit=2.5, q_max_diff=2.0, do_plot=False):
    # Read data
    q_exp, I_exp = read_file(f_exp)
    q_sol, I_sol = read_file(f_sol)

    # Create a regular grid
    q_min = max(q_exp.min(), q_sol.min())
    q_max = min(q_exp.max(), q_sol.max())
    q_regular = np.linspace(q_min, q_max, num_points)

    # Interpolate both functions onto the regular grid
    I_exp_interp = interpolate_data(q_exp, I_exp, q_regular)
    I_sol_interp = interpolate_data(q_sol, I_sol, q_regular)

    # Fit the functions with constraint
    cons = {'type': 'ineq', 'fun': lambda alpha: constraint(alpha, q_regular, I_exp_interp, I_sol_interp, q_min_fit, q_max_fit)}
    result = minimize(fit_function, x0=1.0, args=(q_regular, I_exp_interp, I_sol_interp, q_min_fit, q_max_fit),
                      method='SLSQP', constraints=cons)

    alpha_opt = result.x[0]

    # Compute the difference
    difference = I_exp_interp - alpha_opt * I_sol_interp

    # Check if constraint is satisfied
    min_difference = np.min(difference[(q_regular > q_min_fit) & (q_regular <= q_max_fit)])
    if min_difference < 10.0:
        warnings.warn(f"Constraint not fully satisfied. Minimum difference in fitting range: {min_difference:.2f}")

    # Write the result to a file (only up to q_max_diff)
    mask_diff = q_regular <= q_max_diff
    np.savetxt(output, np.column_stack((q_regular[mask_diff], difference[mask_diff])), fmt='%.6f')

    print(f"Optimal alpha: {alpha_opt}")
    print(f"Optimization success: {result.success}")
    print(f"Optimization message: {result.message}")
    print(f"Minimum difference in fitting range: {min_difference:.2f}")
    print(f"Output written to {output}")

    # Plot if requested
    if do_plot:
        plot_results(q_regular, I_exp_interp, I_sol_interp, alpha_opt, difference, q_min_fit, q_max_fit, q_max_diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit and compare two functions with constraints, allowing negative alpha.")
    parser.add_argument("f_exp", help="File containing experimental data")
    parser.add_argument("f_sol", help="File containing solution data")
    parser.add_argument("output", help="Output file for the difference")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of points in the regular grid")
    parser.add_argument("--q_min_fit", type=float, default=1.7, help="Minimum q value for fitting range")
    parser.add_argument("--q_max_fit", type=float, default=2.5, help="Maximum q value for fitting range")
    parser.add_argument("--q_max_diff", type=float, default=2.0, help="Maximum q value for difference calculation")
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    
    args = parser.parse_args()
    
    main(args.f_exp, args.f_sol, args.output, args.num_points, args.q_min_fit, args.q_max_fit, args.q_max_diff, args.plot)