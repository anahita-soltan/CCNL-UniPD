import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the Parquet files
directory_path = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\all_layers\No_Monotonic'
output_directory = r'C:\Users\Admin\Documents\Mistry\data\Analysis\ANOVA\all_layers\Plots\Count_plots'
excel_output_path = os.path.join(output_directory, 'Nonmono_neuron_analysis_summary.xlsx')

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize a list to store results
results_list = []

# Process each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.parquet'):
        file_path = os.path.join(directory_path, filename)
        data = pd.read_parquet(file_path)

        # Extract model, condition, and layer from the filename
        parts = filename.split('_')
        condition = parts[1].split('_')[0]  # Extract 'adam' or 'pretrained'
        layer = parts[3]  # Extract 'V1', 'V2', 'V4', or 'IT'
    
        neuron_count = data.shape[0]
        
        # Append to results list
        result = {'Condition': condition, 'Layer': layer, 'Count': neuron_count}
        results_list.append(result)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)

# Define neuron totals for each layer to calculate percentage
neuron_totals = {'V1': 200704, 'V2': 200704, 'V4': 200704, 'IT': 50176}

# Calculate overall percentages
results_df['Overall_Percentage'] = results_df['Count'] / results_df['Layer'].map(neuron_totals) * 100

# Save the DataFrame to an Excel file
with pd.ExcelWriter(excel_output_path) as writer:
    results_df.to_excel(writer, index=False, sheet_name='Neuron_Tuning_Analysis')

# Pivot the results to format suitable for plotting
pivot_results = results_df.pivot_table(index='Layer', columns='Condition', values='Overall_Percentage', aggfunc='sum')

# Ensure the correct order for layers and conditions
pivot_results = pivot_results.reindex(['V1', 'V2', 'V4', 'IT'])
pivot_results = pivot_results[['pretrained', 'adam']]  # Make sure to match these to the actual conditions present

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
pivot_results.plot(kind='bar', ax=ax, color=['blue', 'red'])
ax.set_title('Proportion of neurons across layers and conditions')
ax.set_xlabel('Layer')
ax.set_ylabel('Proportion of Neurons (%)')
ax.set_ylim(0, 100)  # Set y-axis to go up to 100%
ax.legend(title='Model')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'neuron_proportion_comparison3.png'))
plt.show()

print(f"Analysis complete. Results saved to {excel_output_path} and plots saved to {output_directory}")
