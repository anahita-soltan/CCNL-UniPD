# ANOVA Analysis of Neuron Activations

This branch contains scripts for conducting ANOVA (Analysis of Variance) on the neuron activations in response to numerosity extracted from different layers and models.

## Overview
The ANOVA analysis compares the neuron counts across different models (Pre-trained and trained using Adam optimizer) both with and without normalization. The goal is to determine statistically significant differences in neuron activations across numbers and conditions in the stimuli.

## Directory Structure
- [ANOVA1](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Anova-analysis/ANOVA1_NonN.py): Python script for running the first ANOVA, namely the analysis outputing "significant" neurons, or neurons that show significantly higher avg activation to numbers (p-value=0.01). This analysis does not acount for the impact of different conditions in number detection.
- [ANOVA2](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Anova-analysis/ANOVA2_NonN.py): Python script for running the second ANOVA, main difference with the first ANOVA is removing the influence of "condition" on neuron activity by discarding those neurons with significant (p-value=0.01) effect for condition and/or  condition/number interaction. 
- **`results/`**: Directory containing the ANOVA results and plots of significant neuron activations.
