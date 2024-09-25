This project was carried out during my internship at the Cognitive Computational Neuroscience Lab (CCNL) at the University of Padova. The focus of the project was to analyze neuron activations and tuning curves in the pre-trained CORnet model, using similar methods described in this Nature article. Additional analyses were performed using data from the same source, available here.

Preprocessing Note:
The pretrained model directory originally labeled epoch1 has been renamed to epoch49 for organizational consistency. However, the data being analyzed still originates from the first epoch (epoch1).

Project Overview
This repository contains various analyses conducted on neuron activations and models. Each branch focuses on a different aspect of the analysis, with further details in the README files within each branch.

Main Branches
main
Contains the core project files and general documentation.

Tsne
Focuses on t-SNE analysis and visualization of neuron activations.

Activation-Extraction
For extraction of neuron activations for exploratory data analysis (EDA).

ANOVA-Analysis
For comparing neuron counts across pre-trained and Adam models using ANOVA.

Tuning-Curves
For calculation and visualization of tuning curves for pooled and normalized neurons.

Nasr-analysis
Includes analysis and modifications based on the Nasr & Neider (2019) method.
