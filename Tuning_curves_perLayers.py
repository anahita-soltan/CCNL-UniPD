import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_gaussian_and_save(df, model_name, layer_name, pn, output_directory):
    file_name = f"{model_name}-{layer_name}-PN{pn}"
    x_labels = df['Number'].values
    data = df[f'PN_{pn}'].values

    try:
        # Fit the Gaussian model to the data
        popt, pcov = curve_fit(gaussian, x_labels, data, p0=[1, x_labels[np.argmax(data)], 1])
        print(f"Fitted parameters for {file_name}: {popt}")

        # Plotting the fitted curve along with the data
        plt.figure()
        plt.plot(x_labels, data, 'ko', label='Normalized Data')
        plt.plot(x_labels, gaussian(x_labels, *popt), 'r-', label='Gaussian Fit')
        plt.title(file_name)
        plt.xlabel('Numerosity')
        plt.ylabel('Normalized Activation')
        plt.legend()
        plt.savefig(f"{output_directory}/{file_name}.png")
        plt.close()

    except Exception as e:
        print(f"Failed to fit Gaussian for {file_name}: {str(e)}")

# Define directories
base_dir = 'C:/Users/Admin/Documents/Mistry/data/Analysis/ANOVA/non_normalized/New/Pooling&Normalizing'
output_dir = 'C:/Users/Admin/Documents/Mistry/data/Analysis/ANOVA/non_normalized/New/GaussianFits'
os.makedirs(output_dir, exist_ok=True)

models = ['cornet_pretrained', 'cornet_adam']
layers = ['V1', 'V2', 'V4', 'IT']

# Loop through each file and fit Gaussian
for model in models:
    for layer in layers:
        file_path = os.path.join(base_dir, f'{model}_epoch49_{layer}_NormalizedPools.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for pn in range(9):  # Assuming PN ranges from 0 to 8
                if f'PN_{pn}' in df.columns:
                    fit_gaussian_and_save(df, model, layer, pn, output_dir)
        else:
            print(f"File not found: {file_path}")
