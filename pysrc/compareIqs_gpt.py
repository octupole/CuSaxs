import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse


def read_data(file_path):
    """Reads the data from the file and returns q and I(q) as numpy arrays."""
    data = np.loadtxt(file_path)
    q = data[:, 0]
    I_q = data[:, 1]
    return q, I_q


def interpolate_data(q, I_q, new_q):
    """Interpolates the data to a new set of q values using cubic splines."""
    spline = interpolate.CubicSpline(q, I_q)
    return spline(new_q)


def fit_alpha(I_exp, I_sol, q):
    """Fits alpha to minimize the difference I_exp(q) - alpha * I_sol(q) with constraints."""

    def objective(alpha):
        diff = I_exp - alpha * I_sol
        # Strong penalty for violating the constraint
        penalty = np.sum((diff[diff <= 10.0] - 10.0) ** 2) * 1e6
        return np.sum(diff**2) + penalty

    result = minimize(objective, x0=1.0)  # Initial guess for alpha
    return result.x[0]


def plot_functions_and_difference(q, I_exp, I_sol, difference, alpha):
    """Plots the experimental, solution functions, and their difference."""
    plt.figure(figsize=(10, 6))

    plt.plot(q, I_exp, label="I_exp(q)", color="blue")
    plt.plot(
        q, alpha * I_sol, label=f"alpha * I_sol(q) (alpha={alpha:.4f})", color="green"
    )
    plt.plot(q, difference, label="Difference", color="red")

    plt.xlim([q.min(), q.max()])
    plt.yscale("log")

    plt.xlabel("q")
    plt.ylabel("I(q) and Difference (log scale)")
    plt.title("Comparison of Experimental and Solution Data with Difference")
    plt.legend()
    plt.grid(True)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process and fit I(q) functions.")
    parser.add_argument("F_exp", type=str, help="Path to the experimental data file.")
    parser.add_argument("F_sol", type=str, help="Path to the solution data file.")
    parser.add_argument(
        "--output",
        type=str,
        default="difference.txt",
        help="Output file name for the difference data.",
    )
    parser.add_argument(
        "--q_min",
        type=float,
        default=0.0,
        help="Minimum q value for interpolation grid.",
    )
    parser.add_argument(
        "--q_max",
        type=float,
        default=2.5,
        help="Maximum q value for interpolation grid.",
    )

    parser.add_argument(
        "--q_step",
        type=float,
        default=0.01,
        help="Step size for the interpolation grid.",
    )

    parser.add_argument(
        "--fit_q_min", type=float, default=1.7, help="Minimum q value for fitting."
    )
    parser.add_argument(
        "--fit_q_max", type=float, default=2.5, help="Maximum q value for fitting."
    )
    parser.add_argument(
        "--q_max_diff",
        type=float,
        default=2.0,
        help="Maximum q value for the difference function output.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the difference and the functions in log scale.",
    )

    args = parser.parse_args()

    # Read the data
    q_exp, I_exp = read_data(args.F_exp)
    q_sol, I_sol = read_data(args.F_sol)

    # Create a regular grid
    q_grid = np.arange(args.q_min, args.q_max + args.q_step, args.q_step)

    # Interpolate both functions onto the same grid
    I_exp_interp = interpolate_data(q_exp, I_exp, q_grid)
    I_sol_interp = interpolate_data(q_sol, I_sol, q_grid)

    # Fit alpha in the user-specified or default range
    fit_mask = (q_grid > args.fit_q_min) & (q_grid <= args.fit_q_max)
    alpha = fit_alpha(I_exp_interp[fit_mask], I_sol_interp[fit_mask], q_grid[fit_mask])
    print(" alpha ", alpha)

    # Compute the difference between the fitted functions within the q_max_diff range
    diff_mask = q_grid <= args.q_max_diff
    difference = I_exp_interp[diff_mask] - alpha * I_sol_interp[diff_mask]

    # Apply the constraint on the difference
    difference[difference < 10.0] = 10.0

    # Save the difference to a file
    output_data = np.column_stack((q_grid[diff_mask], difference))
    np.savetxt(
        args.output, output_data, fmt="%.6f", header="# q Difference", comments=""
    )

    print(f"Alpha fitted value: {alpha}")
    print(f"Difference data saved to {args.output}")

    # Plotting if requested
    if args.plot:
        plot_functions_and_difference(
            q_grid[diff_mask],
            I_exp_interp[diff_mask],
            I_sol_interp[diff_mask],
            difference,
            alpha,
        )


if __name__ == "__main__":
    main()
