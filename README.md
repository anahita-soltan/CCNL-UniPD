# CCNL
The project I worked on during my internship in Cognitive Computational Neuroscience Lab of university of Padova.
It includes analysis of the number neurons in the CORnet model used in https://www.nature.com/articles/s41467-023-39548-5#Sec38 article, using the same method as the one in the article and also some additional analysis.
The Data Source: https://zenodo.org/records/7976287

analyses:
1) t-sne analysis and plots of all layers/models.
2) Some extraction of activations of neurons for the purpose of exploratory DA.
3) ANOVA analysis of pretrained and Adam models, with and without Normalization. Plots of ratios of significant neurons in different layers and between models.
4) Calculation and visualization of Tuning Curves of the pooled & normalized neurons with the same Preferred Number.
5) Using Nasr&Neider(2019) method to remove Summation Units (monotonic neurons) from the dataset, and recalculating Tuning Curves again with Nasr method. 
