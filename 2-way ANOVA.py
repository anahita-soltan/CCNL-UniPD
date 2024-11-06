import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Define the base directory for model data
base_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'

# Path to the directory containing results
results_dir = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Normalized'

# List of models, epochs, and layers
models = ['cornet_pretrained', 'cornet_adam']
epochs = ['epoch49'] 
layers = ['V1', 'V2', 'V4', 'IT']

# Process each model directory and layer
for model in models:
    for ep in epochs:
        for layer in layers:
            # Load significant neurons from previously saved results
            significant_neurons_path = os.path.join(results_dir, f'{model}_{ep}_{layer}_significant_neurons.xlsx')
            if not os.path.exists(significant_neurons_path):
                print(f"No data for {model} {layer} in {ep}")
                continue
            significant_neurons_df = pd.read_excel(significant_neurons_path)

            # File paths for the layer data
            layer_file_path = os.path.join(base_dir, model, ep, f'{layer}.npz')
            label_file_path = os.path.join(base_dir, model, ep, 'label.npz')
            condition_file_path = os.path.join(base_dir, model, ep, 'condition.npz')

            if not all(os.path.exists(path) for path in [layer_file_path, label_file_path, condition_file_path]):
                print(f"Files missing for {model} {layer} in {ep}")
                continue

            # Load data
            layer_data = np.load(layer_file_path)[layer]
            label_data = np.load(label_file_path)['label'].flatten()
            condition_data = np.load(condition_file_path)['condition'].flatten()

            # Analyze each significant neuron for 2-way ANOVA
            results_list = []
            for index, row in significant_neurons_df.iterrows():
                channel = int(row['Channel'])
                kernel = int(row['Kernel'])
                neuron_i = int(row['Neuron_i'])
                neuron_j = int(row['Neuron_j'])

                # Extract activations for this neuron across all conditions
                activations = layer_data[:, channel, kernel, neuron_i, neuron_j]

                # Prepare data for ANOVA
                df_anova = pd.DataFrame({
                    'Activation': activations,
                    'Number': label_data,
                    'Condition': condition_data
                })

                # Perform two-way ANOVA
                anova_model = ols('Activation ~ C(Number) * C(Condition)', data=df_anova).fit()
                anova_results = sm.stats.anova_lm(anova_model, typ=2)

                # Append results
                results_list.append({
                    'Channel': channel,
                    'Kernel': kernel,
                    'Neuron_i': neuron_i,
                    'Neuron_j': neuron_j,
                    'ANOVA_Results': str(anova_results)  # Convert to string to save in Excel
                })

                # Output results
                print(f"ANOVA results for {model} {layer}, Neuron ({channel}, {kernel}, {neuron_i}, {neuron_j}):")
                print(anova_results)

            # Save results to a new file for post-ANOVA analysis
            df_results = pd.DataFrame(results_list)
            new_file_path = os.path.join(results_dir, f'{model}_{ep}_{layer}_two_way_ANOVA_results.xlsx')
            df_results.to_excel(new_file_path, index=False)
            print(f"Saved refined ANOVA results to {new_file_path}")
