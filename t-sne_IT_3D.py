import os
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.io as pio

# Define the directory containing the data
data_dir = '/kaggle/input/mistry-activity/activity/cornet_adam/epoch49/'

# Load data
IPS_data = np.load(os.path.join(data_dir, 'IT.npz'))['IT']
label_data = np.load(os.path.join(data_dir, 'label.npz'))['label'][:, 0]

# Filter out labels for numbers 1 to 9 (including the label "0" representing digit "1")
indices = np.where((label_data >= 0) & (label_data <= 8))[0]
IPS_data_filtered = IPS_data[indices]
label_data_filtered = label_data[indices]

# Reshape data for t-SNE
IPS_data_reshaped = IPS_data_filtered.reshape(IPS_data_filtered.shape[0], -1)

# Perform t-SNE
tsne = TSNE(n_components=3, random_state=42)
tsne_data = tsne.fit_transform(IPS_data_reshaped)

# Create traces for each number
traces = []
for num in range(1, 10):
    indices = np.where(label_data_filtered == num - 1)[0]  # Adjusting for label "0" representing digit "1"
    trace = go.Scatter3d(
        x=tsne_data[indices, 0],
        y=tsne_data[indices, 1],
        z=tsne_data[indices, 2],  # Modify to visualize the third dimension
        mode='markers',
        name=f'Number {num}',
        marker=dict(
            size=5,
            opacity=0.8
        )
    )
    traces.append(trace)

layout = go.Layout(
    title='t-SNE Visualization of Numbers 1 to 9 for IPS Layer (Third Dimension)',
    scene=dict(
        xaxis=dict(title='t-SNE Component 1'),
        yaxis=dict(title='t-SNE Component 2'),
        zaxis=dict(title='t-SNE Component 3')
    )
)

fig = go.Figure(data=traces, layout=layout)

pio.show(fig)
