# t-SNE Analysis

This branch contains code and visualizations related to the t-SNE (t-distributed Stochastic Neighbor Embedding) analysis of neuron activations from different layers and models in the CORnet framework.

## Overview
The purpose of this analysis is to reduce the dimensionality of the neuron activations and cluster similar activations for visualization purposes. We employ t-SNE for both 2D and 3D visualizations, enabling us to observe activation patterns across different layers and models.

## Directory Structure
- **`t-sne.py`**: Main Python script for generating 2D t-SNE visualizations.
- **`t-sne_IT_3D.py`**: Script for performing 3D t-SNE analysis, particularly on Inferior Temporal (IT) neurons.
- **`Tsne_Plots/`**: Directory containing the generated t-SNE plots as PNG images.
