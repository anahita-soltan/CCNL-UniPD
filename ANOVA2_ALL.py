import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Define directories
base_dir = r'C:\Users\Admin\Documents\Mistry\data\activity'
output_dir = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\all_layers\ANOVA2'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the model directories and layers
model_dirs = ['cornet_adam/epoch49']
layers = ['V1', 'V2', 'V4']
significance_level = 0.01

# Process each model and layer
for model_dir in model_dirs:
    model_name = model_dir.split('/')[0]
    epoch = model_dir.split('/')[1]
    
    for layer in layers:
        # Construct paths to the necessary data files
        layer_data_path = os.path.join(base_dir, model_dir, f'{layer}.npz')
        label_data_path = os.path.join(base_dir, model_dir, 'label.npz')
        condition_data_path = os.path.join(base_dir, model_dir, 'condition.npz')

        # Check for file existence
        if not all(os.path.exists(path) for path in [layer_data_path, label_data_path, condition_data_path]):
            print(f"Missing data for {model_dir} {layer}")
            continue

        # Load data
        layer_data = np.load(layer_data_path)[layer]
        label_data = np.load(label_data_path)['label'].flatten()  # Ensure label data is flat
        condition_data = np.load(condition_data_path)['condition'].flatten()  # Ensure condition data is flat

        # Initialize an empty list to store significant neurons
        significant_neurons = []

        # Iterate over all neurons in the layer
        for channel in range(layer_data.shape[1]):
            for kernel in range(layer_data.shape[2]):
                for i in range(layer_data.shape[3]):
                    for j in range(layer_data.shape[4]):
                        
                        # Extract activations for this neuron
                        activations = layer_data[:, channel, kernel, i, j]
                        activations = activations.flatten()  # Make sure activations are flat

                        # Create DataFrame for ANOVA
                        df = pd.DataFrame({
                            'Activation': activations,
                            'Number': label_data,
                            'Condition': condition_data
                        })

                        # Perform two-way ANOVA with interaction
                        model = ols('Activation ~ C(Number) * C(Condition)', data=df).fit()
                        aov_table = sm.stats.anova_lm(model, typ=2)

                        # Check the significance conditions
                        if (aov_table.loc['C(Number)', 'PR(>F)'] < significance_level and
                            aov_table.loc['C(Condition)', 'PR(>F)'] > significance_level and
                            aov_table.loc['C(Number):C(Condition)', 'PR(>F)'] > significance_level):
                            
                            # Calculate the preferred number
                            preferred_number = df.groupby('Number')['Activation'].mean().idxmax()

                            # Save the significant neuron details
                            neuron_dict = {
                                'Channel': channel,
                                'Kernel': kernel,
                                'Neuron_i': i,
                                'Neuron_j': j,
                                'Preferred_Number': preferred_number,
                                'P-Value_Number': aov_table.loc['C(Number)', 'PR(>F)'],
                                'P-Value_Condition': aov_table.loc['C(Condition)', 'PR(>F)'],
                                'P-Value_Interaction': aov_table.loc['C(Number):C(Condition)', 'PR(>F)']
                            }
                            significant_neurons.append(neuron_dict)

        # Save filtered significant neurons to parquet format
        if significant_neurons:
            df_filtered = pd.DataFrame(significant_neurons)
            output_path = os.path.join(output_dir, f'{model_dir.replace("/", "_")}_{layer}_ANOVA2.parquet')
            df_filtered.to_parquet(output_path, index=False)
            print(f"Filtered results saved to {output_path}")
