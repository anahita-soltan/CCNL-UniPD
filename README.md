# Neuron Activation Extraction

This branch contains scripts for extracting neuron activations from different layers and models for exploratory data analysis (EDA). 

## Overview
The purpose of this task is to extract the activations of neurons from the pre-trained CORnet model and other models optimized using Adam and RMSProp. These activations will later be used for statistical comparisons and further analyses such as t-SNE or tuning curve evaluation.

## Directory Structure
- **`activation_extraction.py`**: Main script for extracting neuron activations.
- **`activations/`**: Directory to store the extracted activation data in a structured format for further analysis.
