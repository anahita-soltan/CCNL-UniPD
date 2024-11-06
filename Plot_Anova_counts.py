import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the Excel files
directory_path = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\ANOVA1'
output_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\non_normalized\New\Plots\Count_plots'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize a list to store results
results_list = []

# Process each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx') and not filename.startswith('~$'):
        file_path = os.path.join(directory_path, filename)
        data = pd.read_excel(file_path)

        # Extract model and layer from the filename
        parts = filename.split('_')
        condition = parts[1]  # 'adam' or 'pretrained'
        layer = parts[3].split('.')[0]  # 'V1', 'V2', 'V4', or 'IT'

    
        neuron_count = data.shape[0]

        # Append to results list
        results_list.append({'Condition': condition, 'Layer': layer, 'Count': neuron_count})

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)

# Define neuron totals for each layer to calculate percentage
neuron_totals = {'V1': 200704, 'V2': 200704, 'V4': 200704, 'IT': 50176}

# Calculate percentages
results_df['Percentage'] = results_df.apply(lambda row: (row['Count'] / neuron_totals[row['Layer']]) * 100, axis=1)

# Pivot the results to format suitable for plotting
pivot_results = results_df.pivot_table(index='Layer', columns='Condition', values='Percentage', aggfunc='sum')

# Ensure the correct order for layers and conditions
pivot_results = pivot_results.reindex(['V1', 'V2', 'V4', 'IT'])
pivot_results = pivot_results[['pretrained', 'adam']]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
pivot_results.plot(kind='bar', ax=ax, color=['blue', 'red'])
ax.set_title('Proportion of neurons sensitive to numbers 1-7')
ax.set_xlabel('Layer')
ax.set_ylabel('Proportion of Neurons (%)')
ax.set_ylim(0, 100)  # Set y-axis to go up to 100%
ax.legend(title='Model')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'neuron_proportion_comparison2.png'))
plt.show()
