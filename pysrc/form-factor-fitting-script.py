import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import argparse

def form_factor_with_background(q, delta, delta_rho, background):
    sigma = delta / 4
    return 2 * (delta_rho**2) / (q**2) * (1 - np.cos(q*delta) * np.exp(-q**2 * sigma**2 / 2)) + background

def load_data(filename):
    return np.loadtxt(filename, unpack=True)

def fit_data(q, P):
    initial_guess = [10.0, 100.0, 10000.0]  # initial guess for [delta, delta_rho, background]
    
    # Set bounds for the parameters
    # delta: 10 to 100, delta_rho: 0 to inf, background: 0 to inf
    bounds = ([10, 0, 0], [100, np.inf, np.inf])
    
    popt, pcov = curve_fit(form_factor_with_background, q, P, p0=initial_guess, bounds=bounds)
    return popt, pcov

def plot_results(q, P, popt):
    plt.figure(figsize=(10, 6))
    plt.scatter(q, P, label='Input data')
    q_fit = np.linspace(min(q), max(q), 100)
    plt.plot(q_fit, form_factor_with_background(q_fit, *popt), 'r-', label='Fitted curve')
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('P(q)')
    plt.legend()
    plt.title(f'Form Factor Fitting (δ = {popt[0]:.2f} Å, ΔF = {popt[1]:.2f}, bg = {popt[2]:.4f})')
    plt.show()

def output_fitted(popt, output_file):
    q_out = np.linspace(0.1, 2.0, 1000)
    P_out = form_factor_with_background(q_out, *popt)
    np.savetxt(output_file, np.column_stack((q_out, P_out)), 
               header='q (A^-1)    P(q)', delimiter='    ')

def main():
    parser = argparse.ArgumentParser(description='Fit form factor data with background and constrained bilayer thickness.')
    parser.add_argument('-f', '--file', required=True, help='Input file with q and P(q) data')
    parser.add_argument('--plot', action='store_true', help='Plot input data and fitted curve')
    parser.add_argument('-o', '--output', help='Output file for fitted function')
    args = parser.parse_args()

    # Load data
    q, P = load_data(args.file)

    # Fit data
    popt, pcov = fit_data(q, P)

    # Calculate parameter errors
    perr = np.sqrt(np.diag(pcov))

    # Print results with errors
    print(f'Fitted bilayer thickness (δ): {popt[0]:.2f} ± {perr[0]:.2f} Å')
    print(f'Fitted scattering contrast (ΔF): {popt[1]:.2f} ± {perr[1]:.2f}')
    print(f'Fitted background: {popt[2]:.4f} ± {perr[2]:.4f}')

    # Plot if requested
    if args.plot:
        plot_results(q, P, popt)

    # Output fitted function if requested
    if args.output:
        output_fitted(popt, args.output)

if __name__ == "__main__":
    main()