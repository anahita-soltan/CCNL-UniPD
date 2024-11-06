import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import zscore

# Define directories
base_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'
anova1_result_dir = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\ANOVA1'
output_dir = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\ANOVA2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the model directories and layers
model_dirs = ['cornet_pretrained/epoch49', 'cornet_adam/epoch49']
layers = ['V1', 'V2', 'V4', 'IT']
significance_level = 0.019+-6


# Process each model and layer
for model_dir in model_dirs:
    for layer in layers:
        # Construct paths to the necessary data files
        layer_data_path = os.path.join(base_dir, model_dir, f'{layer}.npz')
        label_data_path = os.path.join(base_dir, model_dir, 'label.npz')
        condition_data_path = os.path.join(base_dir, model_dir, 'condition.npz')
        anova1_results_path = os.path.join(anova1_result_dir, f'{model_dir.replace("/", "_")}_{layer}_ANOVA1.xlsx')

        # Check for file existence
        if not all(os.path.exists(path) for path in [layer_data_path, label_data_path, condition_data_path, anova1_results_path]):
            print(f"Missing data for {model_dir} {layer}")
            continue

        # Load data
        layer_data = np.load(layer_data_path)[layer]
        label_data = np.load(label_data_path)['label'].flatten()  # Ensure label data is flat
        condition_data = np.load(condition_data_path)['condition'].flatten()  # Ensure condition data is flat
        df_anova1 = pd.read_excel(anova1_results_path)

        # Filter neurons based on the first ANOVA results
        significant_neurons = []
        for _, neuron in df_anova1.iterrows():
            channel, kernel, i, j = int(neuron['Channel']), int(neuron['Kernel']), int(neuron['Neuron_i']), int(neuron['Neuron_j'])

            # Extract activations for this neuron
            activations = layer_data[:, channel, kernel, i, j]
            activations = activations.flatten()  # Make sure activations are flat

            # Create DataFrame for ANOVA
            df = pd.DataFrame({
                'Activation': activations,
                'Number': label_data,
                'Condition': condition_data
            })

            # Perform two-way ANOVA
            model = ols('Activation ~ C(Number) + C(Condition)', data=df).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)

            # Check for number and condition significance
            if (aov_table.loc['C(Number)', 'PR(>F)'] < significance_level and
                aov_table.loc['C(Condition)', 'PR(>F)'] > significance_level):
                significant_neurons.append(neuron.to_dict())

        # Save filtered significant neurons
        if significant_neurons:
            df_filtered = pd.DataFrame(significant_neurons)
            output_path = os.path.join(output_dir, f'{model_dir.replace("/", "_")}_{layer}_ANOVA2.xlsx')
            df_filtered.to_excel(output_path, index=False)
            print(f"Filtered results saved to {output_path}")