# README

## Machine Learning Homework 6: Naive Bayes Classification of ICU Patient Survival

This repository contains the code and supplementary materials for the Machine Learning course (Tsinghua University Course 80250993) Homework 6. The task involves classifying patients' survival (0: survived; 1: dead) using 108 features from their Intensive Care Unit (ICU) records.

## Programming Environment

- **Operating System**: [Your Operating System]
- **Python Version**: [Your Python Version]
- **Libraries and Versions**:
  - numpy: [Your numpy Version]
  - pandas: [Your pandas Version]
  - scikit-learn: [Your scikit-learn Version]
  - matplotlib: [Your matplotlib Version]

## Dataset

- **Source**: [Your Dataset Source]
- **Dataset Name**: ICU Patient Dataset
- **Number of Features**: 108
- **Number of Samples**: Training Set - 5000, Test Set - 1097
- **Features**: A mixture of numeric and binary variables such as age, BMI, height, weight, heart rate, blood pressure, etc.

## Experiment Setup

The experiment involves training a Naive Bayes classifier using various feature distributions to classify patient survival. The goal was to find the optimal feature distribution assumptions that maximize the model's predictive performance on the test set. The following feature distributions were considered:

- **Continuous Numeric Features**: Assumed to follow Gaussian distribution.
- **Binary Features**: Assumed to follow Bernoulli distribution.
- **Non-negative Features with Large Range**: Assumed to follow Log-Normal distribution.

## Files

- `train1_icu_data.csv`: Training set feature data.
- `train1_icu_label.csv`: Training set labels.
- `test1_icu_data.csv`: Test set feature data.
- `test1_icu_label.csv`: Test set labels.
- `naive_bayes_classification.ipynb`: Jupyter Notebook containing the Naive Bayes classification code.
- `experiment_report.pdf`: Detailed report of the experiment observations and analysis.

## Experiment Results

The experiment resulted in identifying the optimal feature distributions for the Naive Bayes model. The model achieved a training accuracy of [Your Training Accuracy], a cross-validation accuracy of [Your Cross-Validation Accuracy], and a test set accuracy of [Your Test Accuracy]. The decision risk analysis with a cost-sensitive approach led to a test error rate of [Your Test Error Rate with Decision Risk Analysis].

## Analysis

The analysis of the experiment revealed that the Naive Bayes model performed well in classifying patient survival with a high degree of accuracy. The feature distribution assumptions had a significant impact on the model's performance. The decision risk analysis provided insights into the cost of false positives and false negatives, allowing for a more informed decision-making process in the context of patient survival prediction.

This experiment demonstrates the effectiveness of Naive Bayes classifiers in handling mixed-type data and the importance of选择合适的 feature distributions for optimal model performance. The results can be used to guide clinical decision-making and resource allocation in healthcare settings.

Please ensure that you have the necessary libraries installed to run the code and reproduce the experiments. If you have any questions or require further information, please feel free to reach out.
