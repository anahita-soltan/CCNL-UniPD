import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the Excel files
directory_path = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\ANOVA2'
output_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\ANOVA2\Output'  # Specify the output directory

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx') and not filename.startswith('~$'):
        file_path = os.path.join(directory_path, filename)
        data = pd.read_excel(file_path)

        # Extract model and layer from the filename
        parts = filename.split('_')
        condition = parts[1]  # 'adam' or 'pretrained'
        layer = parts[3]  # 'V1', 'V2', 'V4', or 'IT'

        # Assuming 'Tuned_Number' column exists and contains the tuning data
        if 'Tuned_Number' in data.columns:
            count_data = data['Tuned_Number'].value_counts().reindex(range(9), fill_value=0).rename_axis('Number').reset_index(name='Counts')
            
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.bar(count_data['Number'] + 1, count_data['Counts'], color='b')  # +1 to make numbers 1 to 9
            plt.title(f'Neuron Count for {condition} in {layer}')
            plt.xlabel('Tuned Number')
            plt.ylabel('Counts')
            plt.xticks(ticks=range(1, 10))
            plt.grid(axis='y')

            # Save the plot
            plt.tight_layout()
            output_path = os.path.join(output_directory, f'{condition}_{layer}_neuron_count.png')
            plt.savefig(output_path)
            plt.close()

            print(f'Saved plot to: {output_path}')
