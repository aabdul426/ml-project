# ml‑project

This repository contains the source code and report for a machine‑learning classification project completed for ITCS 3156 (Machine Learning). The goal of the project is to build a complete pipeline—from data generation and exploration through model training and evaluation—using a synthetic dataset.

## Overview

- **Dataset:** A synthetic classification dataset of 12 000 examples with 8 features, generated via `scikit‑learn`’s `make_classification` function.
- **Models:** Logistic Regression and Random Forest.
- **Evaluation:** Accuracy, precision, recall, F1 score, and ROC/AUC on a held‑out test set; 5‑fold cross‑validation for the baseline model.
- **Outputs:** The Python script generates a detailed PDF report summarizing data exploration, preprocessing, model setup, results and conclusions.

## Files

- `project_analysis.py` – Script that generates the dataset, performs the analysis, creates plots, and compiles the final report.  
- `report.pdf` – The generated report describing the project (can be uploaded for grading).  
- `README.md` – This file.

## Requirements

- Python 3.8+  
- Packages: `numpy`, `pandas`, `scikit‑learn`, `matplotlib`, `seaborn`

You can install the required packages with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
