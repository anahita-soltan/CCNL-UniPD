import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Define directories
input_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Final\ANOVA1'
output_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Final\ANOVA2'
data_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Set the significance level
significance_level = 0.01

# List the models and layers you are analyzing
models = ['cornet_pretrained', 'cornet_adam']
epochs = ['epoch49']
layers = ['V1', 'V2', 'V4', 'IT']

# Iterate through each model, epoch, and layer
for model_name in models:
    for epoch in epochs:
        for layer in layers:
            input_file = f'{model_name}_{epoch}_{layer}_ANOVA1.xlsx'
            file_path = os.path.join(input_directory, input_file)

            # Check if the file exists
            if not os.path.exists(file_path):
                print(f'File not found: {file_path}')
                continue

            # Load the significant neurons data
            df_neurons = pd.read_excel(file_path)
            results = []

            # Load activity and condition data
            activity_path = os.path.join(data_dir, model_name, epoch, f'{layer}.npz')
            label_path = os.path.join(data_dir, model_name, epoch, 'label.npz')
            condition_path = os.path.join(data_dir, model_name, epoch, 'condition.npz')

            if not all(os.path.exists(path) for path in [activity_path, label_path, condition_path]):
                print(f"Data files missing for {model_name} {layer} in {epoch}")
                continue

            layer_data = np.load(activity_path)[layer]
            label_data = np.load(label_path)['label'].flatten()
            condition_data = np.load(condition_path)['condition'].flatten()

            # Analyze each neuron
            for _, neuron in df_neurons.iterrows():
                channel = int(neuron['Channel'])
                kernel = int(neuron['Kernel'])
                neuron_i = int(neuron['Neuron_i'])
                neuron_j = int(neuron['Neuron_j'])

                # Debugging output
                print(f"Channel: {channel}, Kernel: {kernel}, Neuron_i: {neuron_i}, Neuron_j: {neuron_j}")
                print(f"Layer data shape: {layer_data.shape}")

                # Check index validity
                if channel >= layer_data.shape[1] or kernel >= layer_data.shape[2] or neuron_i >= layer_data.shape[3] or neuron_j >= layer_data.shape[4]:
                    print("Index out of bounds error!")
                    continue

                # Extract activations for this neuron
                try:
                    activations = layer_data[:, channel, kernel, neuron_i, neuron_j]
                except IndexError as e:
                    print(f"Indexing error: {e}")
                    continue
                # Prepare DataFrame for 2-way ANOVA
                df_anova = pd.DataFrame({
                    'Activation': activations,
                    'Number': label_data,
                    'Condition': condition_data
                })

                # Conduct 2-way ANOVA
                anova_model = ols('Activation ~ C(Number) * C(Condition)', data=df_anova).fit()
                anova_results = sm.stats.anova_lm(anova_model, typ=2)

                p_cond = anova_results.loc['C(Condition)', 'PR(>F)']
                p_inter = anova_results.loc['C(Number):C(Condition)', 'PR(>F)']

                if p_cond > significance_level and p_inter > significance_level:
                    results.append({
                        'Channel': channel,
                        'Kernel': kernel,
                        'Neuron_i': neuron_i,
                        'Neuron_j': neuron_j,
                        'p_value_Condition': p_cond,
                        'p_value_Interaction': p_inter
                    })

            output_file = f'{model_name}_{epoch}_{layer}_filtered.xlsx'
            output_path = os.path.join(output_directory, output_file)

            try:
                pd.DataFrame(results).to_excel(output_path, index=False)
                print(f'Saved filtered ANOVA results to {output_path}')
            except Exception as e:
                print("Failed to save file:", e)
