import os
import pandas as pd

# Define directories
anova1_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Final\ANOVA1'
anova2_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\Final\ANOVA2'

# List the models, epochs, and layers you are analyzing
models = ['cornet_pretrained', 'cornet_adam']
epochs = ['epoch49']
layers = ['V1', 'V2', 'V4', 'IT']

# Function to create a mapping from the ANOVA1 files
def create_tuned_number_mapping():
    tuned_number_mapping = {}
    for model_name in models:
        for epoch in epochs:
            for layer in layers:
                input_file = f'{model_name}_{epoch}_{layer}_ANOVA1.xlsx'
                file_path = os.path.join(anova1_directory, input_file)
                
                # Check if the file exists
                if not os.path.exists(file_path):
                    print(f'File not found: {file_path}')
                    continue
                
                # Load the significant neurons data
                df_neurons = pd.read_excel(file_path)
                
                # Create a mapping for the "Tuned Number"
                for _, neuron in df_neurons.iterrows():
                    key = (model_name, epoch, layer, int(neuron['Channel']), int(neuron['Kernel']), int(neuron['Neuron_i']), int(neuron['Neuron_j']))
                    tuned_number_mapping[key] = neuron['Tuned_Number']
    return tuned_number_mapping

# Function to update the ANOVA2 files with the "Tuned Number" column
def update_anova2_files(tuned_number_mapping):
    for model_name in models:
        for epoch in epochs:
            for layer in layers:
                input_file = f'{model_name}_{epoch}_{layer}_filtered.xlsx'
                file_path = os.path.join(anova2_directory, input_file)
                
                # Check if the file exists
                if not os.path.exists(file_path):
                    print(f'File not found: {file_path}')
                    continue
                
                # Load the filtered neurons data
                df_filtered = pd.read_excel(file_path)
                
                # Add the "Tuned Number" column
                tuned_numbers = []
                for _, neuron in df_filtered.iterrows():
                    key = (model_name, epoch, layer, int(neuron['Channel']), int(neuron['Kernel']), int(neuron['Neuron_i']), int(neuron['Neuron_j']))
                    tuned_number = tuned_number_mapping.get(key, None)
                    tuned_numbers.append(tuned_number)
                
                df_filtered['Tuned_Number'] = tuned_numbers
                
                # Save the updated file
                output_file = f'{model_name}_{epoch}_{layer}_ANOVA2.xlsx'
                output_path = os.path.join(anova2_directory, output_file)
                
                try:
                    df_filtered.to_excel(output_path, index=False)
                    print(f'Saved updated ANOVA results to {output_path}')
                except Exception as e:
                    print("Failed to save file:", e)

# Create the tuned number mapping from ANOVA1 files
tuned_number_mapping = create_tuned_number_mapping()

# Update the ANOVA2 files with the "Tuned Number" column
update_anova2_files(tuned_number_mapping)
