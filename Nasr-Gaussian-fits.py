import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

def gaussian(x, a, b, c):
    return a * np.exp(-0.5 * ((x - b) / c) ** 2)

# Adjust this function to calculate the goodness of fit and sigma values
def fit_gaussian(data, scale_function):
    x_data = scale_function(np.arange(len(data)))
    y_data = data
    try:
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0=[1, np.argmax(y_data), 1])
        residuals = y_data - gaussian(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, r_squared
    except Exception as e:
        print(f"Failed to fit: {e}")
        return None, None

# Generate plots
def generate_plots(dir_path):
    # Storage for results
    results = {
        "Linear": [],
        "Power 0.5": [],
        "Power 0.33": [],
        "Log": []
    }

    scales = {
        "Linear": lambda x: x,
        "Power 0.5": lambda x: x**0.5,
        "Power 0.33": lambda x: x**0.33,
        "Log": lambda x: np.log2(x + 1)
    }

    # Process each file
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(dir_path, file))
            for column in df.columns[1:]:  # Skip 'Number' column
                data = df[column].values
                for name, func in scales.items():
                    popt, r_squared = fit_gaussian(data, func)
                    if popt is not None:
                        results[name].append((popt[2], r_squared))  # Store sigma and R^2

    # Plotting Goodness of Fit
    fig, ax = plt.subplots()
    for scale, values in results.items():
        sigmas, r_squares = zip(*values)
        avg_r = np.mean(r_squares)
        std_r = np.std(r_squares)
        ax.bar(scale, avg_r, yerr=std_r, label=f'{scale} (avg R²)')

    ax.set_ylabel('Average R²')
    ax.set_title('Goodness of Fit (R²) by Scale')
    plt.show()

    # Plotting Sigma Values
    fig, ax = plt.subplots()
    for scale, values in results.items():
        sigmas, r_squares = zip(*values)
        numbers = range(1, len(sigmas) + 1)
        scatter = ax.scatter(numbers, sigmas, label=f'{scale} scale')

         # Fit a line and plot without adding to legend
        slope, intercept, r_value, p_value, std_err = linregress(numbers, sigmas)
        ax.plot(numbers, intercept + slope*np.array(numbers), color=scatter.get_facecolor()[0])

    ax.set_xlabel('Number')
    ax.set_ylabel('Sigma of Gaussian Fit')
    ax.set_title('Sigma of Gaussian Fit by Scale')
    ax.legend()
    plt.show()

# Directory path
dir_path = 'C:\\Users\\Admin\\Documents\\Mistry\\data\\Analysis\\ANOVA\\non_normalized\\New\\Pooling&Normalizing'
generate_plots(dir_path)
