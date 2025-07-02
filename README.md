# Diabetes Prediction Model

This repository contains a machine learning pipeline for predicting diabetes based on health indicators.

## Improvements to the Original Code

The original code (`code1.py`) has been significantly improved to address several issues and enhance functionality:

### 1. Robust Error Handling
- Added comprehensive error handling for file operations, data loading, and model training
- Graceful handling of missing files, columns, and other potential runtime errors

### 2. Modular Code Structure
- Refactored code into well-defined functions with clear responsibilities
- Added docstrings and comments for better code understanding
- Improved code organization following a logical workflow

### 3. Data Analysis and Validation
- Added data validation to check for required columns
- Implemented analysis of class distribution and detection of class imbalance
- Added correlation analysis to identify potential spurious relationships

### 4. Feature Selection
- Implemented feature selection to identify the most relevant predictors
- Used multiple methods (statistical tests and model-based importance)
- Visualization of feature importance for better interpretability

### 5. Class Imbalance Handling
- Detection and handling of class imbalance using class weights
- Stratified sampling to maintain class distribution in train/test splits

### 6. Comprehensive Model Evaluation
- Added multiple evaluation metrics (accuracy, precision, recall, F1-score)
- Implemented confusion matrix analysis
- Added ROC-AUC for binary classification performance
- Implemented cross-validation for more robust performance assessment

### 7. Model Persistence
- Added functionality to save the trained model for future use
- Created proper directory structure for model storage

### 8. Visualization
- Added visualization of feature correlations
- Implemented feature importance visualization

## Usage

To run the improved diabetes prediction model:

```bash
python code1.py
```

The script will:
1. Load and analyze the diabetes indicator dataset
2. Perform feature selection
3. Train a Random Forest classifier
4. Evaluate model performance
5. Save the trained model

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
- joblib