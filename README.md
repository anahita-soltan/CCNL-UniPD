# Nasr, Viswanathan & Neider (2019) Method Application

This branch implements the method described in Nasr & Neider (2019) to remove summation units (monotonic neurons) from the dataset and recalculates tuning curves after this refinement.
[paper](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://homepages.uni-tuebingen.de/andreas.nieder/Nasr,%20Viswanathan,%20Nieder%20(2019)%20SciAdv.pdf)

## Overview
This analysis uses the method described by Nasr & Nieder (2019) to refine and analyze neural activity data from the CORnet model. We first identify and remove monotonic neurons, known as summation units, which respond in a linearly increasing or decreasing manner to stimulus magnitude without selectivity. This filtering process leaves a dataset of neurons with non-monotonic responses, more relevant for understanding stimulus-specific tuning.

Following the removal of summation units, we conduct pooling and normalization of neural responses based on their preferred stimulus number. Finally, we apply Gaussian fits to the normalized tuning curves to quantify the neurons' selective response characteristics.

## Why This Is Useful

**Enhances Specificity of Tuning:** Removing summation units focuses our analysis on neurons with non-monotonic, stimulus-selective responses, which are key for understanding specific numerical encoding.
**Improves Dataset Quality:** Normalizing and pooling neuron responses across conditions ensures comparability and reduces variability, while Gaussian fitting allows us to quantify the degree and sharpness of selectivity in these responses.
**Enables Comparisons Across Models:** The cleaned and refined data can be used to assess differences in tuning precision across CORnet model variants (e.g., cornet_pretrained and cornet_adam), contributing to our understanding of neural representations in artificial networks.
Methodology

## The analysis is conducted in three main steps:

**Removing Monotonic (Summation) Neurons:**
Each neuron’s response to varying stimulus magnitudes is assessed via linear regression.
Neurons exhibiting a high linear correlation (e.g., 𝑅2 > 0.5) are marked as monotonic and excluded from further analysis.
Only neurons tuned to the stimulus extremes (labels 0 or 8) are evaluated for monotonicity, focusing on those most likely to show non-specific response patterns.

**Pooling and Normalizing Neuron Activations:**
For each preferred numerosity, neurons are pooled, and their responses are averaged across stimulus presentations.
Normalization is applied to bring activations into a common scale, allowing for fair comparisons. Activation values are scaled between the minimum and maximum observed responses.
This step results in a set of normalized tuning curves for each preferred number, ready for fitting.

**Gaussian Fitting of Tuning Curves:**
Gaussian curves are fitted to the normalized tuning curves to quantify tuning selectivity.
Different scaling transformations (linear, power, and logarithmic) are applied to assess which provides the best fit.
Sigma values and goodness-of-fit (R²) are computed for each Gaussian fit, providing insight into the sharpness and reliability of tuning for each scale.
These values are visualized to compare tuning characteristics across different transformations, highlighting the effect of scaling on model selectivity.

## Directory Structure
[Nasr-Removing-Summation.py](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Nasr-analysis/Nasr-Removing-Summation.py) Script for identifying and removing monotonic (summation) neurons based on linear regression analysis of their activity patterns.

[Nasr-Pooling&Normalization.py](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Nasr-analysis/Nasr-Pooling%26Normalization.py) Script for pooling and normalizing neuron activations by preferred numerosity, creating standardized tuning curves for each neuron group.

[Nasr-Gaussian-fits.py](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Nasr-analysis/Nasr-Gaussian-fits.py) Script for applying Gaussian fits to the normalized tuning curves, with different scaling transformations and visualizations of fit quality and selectivity (sigma values).

recalculated_tuning_curves/: Directory where the outputs from the tuning curve analysis, including normalized data and Gaussian fits, are stored as .csv files
