# Predictive Maintenance Classifier

A machine learning project to predict industrial machine failures 
using sensor data — built as my first end-to-end ML project.

## Problem
Unplanned machine failures cost manufacturers millions. 
This project builds a classifier to predict failure before it happens 
using real sensor readings.

## Dataset
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)  
10,000 rows | 14 features | Binary classification

## What I did
- Exploratory Data Analysis — sensor distributions, correlation heatmap, pairplots, boxplots
- Feature Engineering — Power (Torque × RPM), Temperature delta
- Trained Random Forest and XGBoost classifiers
- Evaluated with F1-score, confusion matrix, classification report
- Feature importance analysis

## Results
| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Random Forest | 1.00 | 0.97 | 0.99 |
| XGBoost | 1.00 | 0.97 | 0.99 |

Cross-validation F1: **0.987**

## Key Finding
Top predictive features: HDF, OSF, PWF, Power, TWF  
Limitation: failure type flags may not be available in real-time deployment.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Random Forest
