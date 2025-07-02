#!/usr/bin/env python3
"""
Adult Income Prediction Model

This script implements a machine learning pipeline to predict income levels
from the Adult dataset using logistic regression.
"""

import pandas as pd
import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import class_weight


def get_project_root():
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('adult_income_model.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train income prediction model')
    parser.add_argument('--data-path', type=str, help='Path to the dataset')
    parser.add_argument('--test-size', type=float, default=0.2, 
                      help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42, 
                      help='Random seed for reproducibility')
    parser.add_argument('--model-output', type=str, default='models/adult_income_model.joblib',
                      help='Path to save the trained model')
    parser.add_argument('--tune-hyperparams', action='store_true',
                      help='Perform hyperparameter tuning')
    return parser.parse_args()


def load_data(data_path=None):
    """
    Load the Adult dataset from the specified path or download it if not available.
    
    Args:
        data_path: Path to the CSV file containing the dataset
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    if data_path and os.path.exists(data_path):
        try:
            logger.info(f"Loading data from {data_path}")
            return pd.read_csv(data_path)
        except pd.errors.ParserError:
            logger.error(f"Error parsing {data_path}. Check file format.")
            sys.exit(1)
    else:
        logger.info("Data file not found. Downloading from UCI repository...")
        try:
            from sklearn.datasets import fetch_openml
            adult = fetch_openml(name='adult', version=1, as_frame=True)
            data = adult.data
            data['salary'] = adult.target
            logger.info("Data downloaded successfully")
            return data
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            sys.exit(1)


def explore_data(data, target_col):
    """
    Perform basic exploratory data analysis.
    
    Args:
        data: The dataset to explore
        target_col: The name of the target column
        
    Returns:
        pandas.Series: Distribution of the target variable
    """
    logger.info(f"Dataset shape: {data.shape}")
    
    logger.info("Missing values per column:")
    missing_values = data.isnull().sum()
    logger.info(f"\n{missing_values}")
    
    logger.info("Target distribution:")
    target_counts = data[target_col].value_counts(normalize=True)
    logger.info(f"\n{target_counts}")
    
    # For numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        logger.info("Numeric column statistics:")
        logger.info(f"\n{data[numeric_cols].describe()}")
    
    return target_counts


def create_preprocessor(numeric_cols, categorical_cols):
    """
    Create a column transformer for preprocessing features.
    
    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        ColumnTransformer: The preprocessing pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor


def train_model(X_train, y_train, preprocessor, tune_hyperparams=False):
    """
    Train a logistic regression model with the given data.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Feature preprocessing pipeline
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        Pipeline: The trained model pipeline
    """
    # Compute class weights to handle imbalance
    class_weights = class_weight.compute_class_weight('balanced', 
                                                     classes=np.unique(y_train), 
                                                     y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight=class_weight_dict, max_iter=1000))
    ])
    
    if tune_hyperparams:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        logger.info("Training model with default parameters...")
        model.fit(X_train, y_train)
        return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model and print performance metrics.
    
    Args:
        model: The trained model
        X_test: Test features
        y_test: Test target
        label_encoder: Encoder used for the target variable
        
    Returns:
        dict: Classification report as a dictionary
    """
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                 target_names=label_encoder.classes_,
                                 output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, 
                                    target_names=label_encoder.classes_))
    
    return report


def plot_feature_importance(model, preprocessor, feature_names, output_file='feature_importance.png'):
    """
    Plot feature importance for the trained model.
    
    Args:
        model: The trained model (must have coef_ attribute)
        preprocessor: The preprocessing pipeline
        feature_names: Original feature names
        output_file: Path to save the plot
    """
    if hasattr(model, 'coef_'):
        # For linear models
        feature_importance = model.coef_[0]
        # Get feature names after preprocessing
        all_feature_names = preprocessor.get_feature_names_out(feature_names)
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': np.abs(feature_importance)
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
        plt.xlabel('Absolute Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_file}")


def save_model(model, output_path):
    """
    Save the trained model to disk.
    
    Args:
        model: The trained model
        output_path: Path to save the model
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Define data paths
    project_root = get_project_root()
    data_path = args.data_path
    if not data_path:
        data_path = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
    
    # Load and explore data
    data = load_data(data_path)
    
    # Define features
    numeric_columns = ['age', 'hours-per-week']
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                          'relationship', 'race', 'sex', 'native-country']
    target = 'salary'
    
    # Validate columns
    expected_columns = numeric_columns + categorical_columns + [target]
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing expected columns: {missing_columns}")
        sys.exit(1)
    
    # Explore data
    explore_data(data, target)
    
    # Prepare data
    X = data[numeric_columns + categorical_columns]
    y = data[target]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, random_state=args.random_state
    )
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_columns, categorical_columns)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor, args.tune_hyperparams)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Plot feature importance
    plot_feature_importance(
        model.named_steps['classifier'], 
        model.named_steps['preprocessor'],
        numeric_columns + categorical_columns
    )
    
    # Save model
    save_model(model, args.model_output)


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting Adult Income Prediction pipeline")
    
    try:
        main()
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)