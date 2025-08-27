import numpy as np
import argparse
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Define the lamellar model as described in SasView
def lamellar_model(q, scale, spacing, delta_rho, sigma, background):
    """
    Lamellar form factor model as described by SasView.

    Parameters:
    q : array
        Scattering vector values (1/Angstrom)
    scale : float
        Scale factor (intensity)
    spacing : float
        Lamellar spacing (d-spacing between bilayers, in Angstroms)
    delta_rho : float
        Scattering length density contrast (difference between materials)
    sigma : float
        Roughness of the interface (standard deviation of the Gaussian)
    background : float
        Background scattering intensity

    Returns:
    I(q) : array
        Scattered intensity at each q value
    """
    Pq = (2 * delta_rho * np.sin(q * spacing / 2) / (q * spacing)) ** 2
    Sq = np.exp(-(sigma**2) * q**2 / 2)
    Iq = scale * Pq * Sq + background
    return Iq


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fit SAXS data to the lamellar model.")
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Input ASCII file with experimental data (q, I(q))",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the experimental data and fitted model",
    )
    parser.add_argument(
        "--output",
        default="fitted_data.txt",
        help="Output file for saving the fitted data",
    )
    return parser.parse_args()


# Load data from ASCII file
def load_data(file):
    data = np.loadtxt(file)
    q_data = data[:, 0]  # First column is q
    intensity_data = data[:, 1]  # Second column is intensity
    return q_data, intensity_data


# Main fitting function
def fit_lamellar_model(q_data, intensity_data):
    # Initial guess for fitting parameters: [scale, spacing, delta_rho, sigma, background]
    initial_guess = [1, 10, 1, 2, 0]

    # Perform the curve fitting
    popt, pcov = curve_fit(lamellar_model, q_data, intensity_data, p0=initial_guess)

    # Extracting the optimal parameters
    scale_fit, spacing_fit, delta_rho_fit, sigma_fit, background_fit = popt

    # Print fitted parameters
    print(f"Fitted scale: {scale_fit}")
    print(f"Fitted spacing: {spacing_fit} Å")
    print(f"Fitted delta_rho: {delta_rho_fit}")
    print(f"Fitted sigma: {sigma_fit} Å")
    print(f"Fitted background: {background_fit}")

    return popt


# Write the fitted function to a file
def write_fitted_data(filename, popt):
    q_fitted = np.linspace(0.1, 2.0, 500)  # q values from 0.1 to 2.0 A⁻¹
    fitted_intensity = lamellar_model(q_fitted, *popt)

    # Save to file
    np.savetxt(
        filename,
        np.column_stack((q_fitted, fitted_intensity)),
        header="q(A^-1) Fitted_I(q)",
    )


# Plot function
def plot_comparison(q_data, intensity_data, popt):
    fitted_intensity = lamellar_model(q_data, *popt)

    plt.figure()
    plt.plot(q_data, intensity_data, "bo", label="Experimental Data")
    plt.plot(q_data, fitted_intensity, "r-", label="Fitted Model")
    plt.xlabel("q (1/Å)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.title("Lamellar Model Fitting")
    plt.show()


# Main script
if __name__ == "__main__":
    args = parse_args()

    # Load the data from file
    q_data, intensity_data = load_data(args.file)

    # Fit the model to the data
    popt = fit_lamellar_model(q_data, intensity_data)

    # Optionally plot the comparison
    if args.plot:
        plot_comparison(q_data, intensity_data, popt)

    # Write the fitted data to an output file
    write_fitted_data(args.output, popt)
