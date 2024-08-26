import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Import from scikit-learn instead of cuml

# Define the directory containing the data
data_dir = '/kaggle/input/mistry-activity/activity/'

# List of model directories
model_directories = ['cornet_pretrained/epoch49', 'cornet_adam/epoch49',
                     'cornet_rmsprop/epoch49', 'cornet_sgd/epoch49']
# Define layer names
layers = ['V1', 'V2', 'V4', 'IT']

# Load data and create t-SNE plots for each model and layer
for model_dir in model_directories:
    for layer in layers:
        # Load data
        layer_data = np.load(os.path.join(data_dir, model_dir, f'{layer}.npz'))[layer]
        label_data = np.load(os.path.join(data_dir, model_dir, 'label.npz'))['label'][:, 0]

        # Reshape data for t-SNE
        layer_data_reshaped = layer_data.reshape(layer_data.shape[0], -1)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, method='barnes_hut')  # Adjusted for sklearn
        tsne_data = tsne.fit_transform(layer_data_reshaped)
        
        # Plot t-SNE results
        plt.figure(figsize=(10, 8))
        for i in range(9):  # Assuming 9 classes (numbers 1 to 9)
            indices = np.where(label_data == i)[0]
            plt.scatter(tsne_data[indices, 0], tsne_data[indices, 1], label=f'Number {i + 1}')  # Correctly labeling as 1-9
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f't-SNE Visualization of {layer} in {model_dir}')
        plt.legend()
        plt.show()
        
        # Create directory to save the plots if it doesn't exist
        save_dir = os.path.join('tsne_plots', model_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'{layer}_tsne.png'))
        plt.close()  # Close the plot to avoid displaying multiple plots
        
        # After performing t-SNE
        component_1 = tsne_data[:, 0]
        component_2 = tsne_data[:, 1]

        # Visualize the components
        plt.figure(figsize=(8, 6))
        plt.scatter(component_1, component_2, c=label_data)  # Color-coded by class labels
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f't-SNE Components for {layer} in {model_dir}')
        plt.colorbar(label='Class Label')
        plt.savefig(os.path.join(save_dir, f'{layer}_tsne_components.png'))
        plt.close()  # Close the plot
