# Laughter Authenticity Classifier
---
---
## Copyright © 2023 ~ MIT License
---
---

This repository contains the Python code for a Support Vector Machine (SVM) trained to determine the authenticity of laughter based on its acoustic features.
The classifier functions as a demonstrator for my thesis for the MSc Voice Technology at the University of Groningen:
"*The relevance of using authentic laughter data in natural laughter synthesis; A case study on LaughNet*" (2022/2023)

## Code description
The code consists of 5 parts:
1. Data preparation
2. Training the classifier
3. Evaluating the classifier
4. Factor analysis
5. Error analysis

In the data preparation, the acoustic features are extracted from the laughter, rid of outliers, and consecutively normalised.
In the training, the data is split into a train and test set, and the hyperparameters of the classifier are fine-tuned using grid search.
In the evaluation, confusion matrices and histograms of projections are created to evaluate the performance of the classifier.
In the factor analysis, it is checked which combinations of acoustic features contributed the most to the decision boundary.
In the error analysis, the acoustic features of misclassified laughter samples are plotted against the respective class means of acted and spontaneous laughter, which can then be cross-referenced with the factor analysis.

Lastly, the sources used in the writing of this code have been listed at the bottom of the code.
