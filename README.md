This project was carried out during my internship at the Cognitive Computational Neuroscience Lab (CCNL) at the University of Padova. The focus of the project was to analyze neuron activations and tuning curves in the pre-trained CORnet model, using similar methods described in this Nature article. Additional analyses were performed using data from the same source, available here.

### Preprocessing Note:
The pretrained model directory originally labeled epoch1 has been renamed to epoch49 for organizational consistency. However, the data being analyzed still originates from the first epoch (epoch1).

### Project Overview
This repository contains various analyses conducted on neuron activations and models. Each branch focuses on a different aspect of the analysis, with further details in the README files within each branch.

### Branches

1. [Tsne](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/tree/Tsne)  
   Focuses on t-SNE analysis and visualization of neuron activations.

2. [Activation-Extraction](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/tree/Activation-Extraction)  
   For extraction of neuron activations for exploratory data analysis (EDA).

3. [ANOVA-Analysis]([https://github.com/anahita-soltan/CCNL/tree/ANOVA-Analysis](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/tree/Anova-analysis)  
   For comparing neuron counts across pre-trained and Adam models using ANOVA.

4. [Tuning-Curves](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/tree/Tuning-Curves)  
   For calculation and visualization of tuning curves for pooled and normalized neurons.

5. [Nasr-analysis](https://github.com/anahita-soltan/CCNL-Cognitive_Computational_Neuroscience_Lab/tree/Nasr-analysis)  
   Includes analysis and modifications based on the Nasr & Neider (2019) method.
