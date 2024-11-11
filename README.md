# Tuning Curve Analysis

This branch focuses on calculating and visualizing tuning curves for pooled and normalized neurons based on their preferred numerosity, following methods detailed in the Mistry paper. These tuning curves capture the response properties of neurons across various experimental conditions, offering insights into numerosity representation in the CORnet model prior to implementing the Nasr model. For access to final results refer to 'Tuning Curves' and 'Plots' folders in
[Final Results](https://unipdit-my.sharepoint.com/:u:/g/personal/anahita_soltantouyeh_studenti_unipd_it/ESQ4FACSGfhEu2Qlz6f2fdgBfKnVg_-AuJR0_Z10cjYK7Q?e=OR3KvO)

## Overview

Tuning curves are used to characterize the responsiveness of neurons to specific stimulus magnitudes, helping to identify numerosity-selective neurons. In this analysis, we:
- **Pool Neurons by Preferred Numerosity**: Aggregate neurons based on their numerosity preference as determined by ANOVA.
- **Normalize Responses**: Scale neuron activations to a common range to facilitate comparison.
- **Generate Tuning Curves**: Create and plot tuning curves to observe peak responses and response spread for each numerosity group, providing a baseline measurement before further refinement with the Nasr method.

## Directory Structure

- **[All_Tuning_Curves_&_Plots.py](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Tuning-curves/All_Tuning_Curves_%26_Plots.py)**: Final script used for generating comprehensive tuning curve plots across layers and numerosities after the latest version of ANOVA.
- **[Tuning_curve.py](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Tuning-curves/Tuning_curve.py)**: Preliminary script for calculating initial tuning curves based on preferred numerosity.
- **[Tuning_curves_perLayers.py](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Tuning-curves/Tuning_curves_perLayers.py)**: Script for calculating and visualizing tuning curves for each layer individually, allowing for layer-specific insights.
- **[tuning_precision_results.csv](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/blob/Tuning-curves/tuning_precision_results.csv)**: Contains preliminary tuning precision metrics (e.g., peak response, spread) for each numerosity.
