# ANOVA Analysis of Neuron Activations

This branch contains scripts for conducting ANOVA (Analysis of Variance) on the neuron activations in response to numerosity extracted from different layers and models.

## Overview
The ANOVA analysis compares the neuron counts across different models (Pre-trained and trained using Adam optimizer) both with and without normalization. The goal is to determine statistically significant differences in neuron activations across numbers and conditions in the stimuli.

## Directory Structure
- **`anova_analysis.py`**: Main Python script for running the ANOVA analysis.
- **`results/`**: Directory containing the ANOVA results and plots of significant neuron activations.
