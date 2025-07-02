# Bugs and Improvements for code2.py

## Identified Bugs

1. **Missing `utils.py` Module**
   - **Bug**: The script imports `get_project_root` from a non-existent utils module.
   - **Fix**: Create a utils.py file with the required function:
     ```python
     # utils.py
     import os
     
     def get_project_root():
         """Return the absolute path to the project root directory."""
         return os.path.dirname(os.path.abspath(__file__))
     ```

2. **Missing Data Directory**
   - **Bug**: The script attempts to load data from a non-existent "datasets/adult_data" directory.
   - **Fix**: Create the directory structure and download the dataset, or modify the path to point to the actual location:
     ```python
     # Add error handling and alternative data sources
     raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
     if not os.path.exists(raw_data_file):
         # Alternative: Download from UCI repository
         print(f"Data file not found at {raw_data_file}. Downloading from UCI repository...")
         from sklearn.datasets import fetch_openml
         adult = fetch_openml(name='adult', version=1, as_frame=True)
         data = adult.data
         data['salary'] = adult.target
     else:
         data = pd.read_csv(raw_data_file)
     ```

3. **No Error Handling**
   - **Bug**: The script will crash if files are missing or data has unexpected structure.
   - **Fix**: Add try-except blocks around critical operations:
     ```python
     try:
         data = pd.read_csv(raw_data_file)
     except FileNotFoundError:
         print(f"Error: Data file not found at {raw_data_file}")
         sys.exit(1)
     except pd.errors.ParserError:
         print(f"Error: Unable to parse {raw_data_file}. Check file format.")
         sys.exit(1)
     ```

4. **Potential Column Mismatch**
   - **Bug**: The script assumes specific column names without verifying their existence.
   - **Fix**: Add validation for expected columns:
     ```python
     expected_columns = numeric_columns + categorical_columns + [target]
     missing_columns = [col for col in expected_columns if col not in data.columns]
     if missing_columns:
         print(f"Error: Missing expected columns: {missing_columns}")
         sys.exit(1)
     ```

## Areas for Improvement

1. **Add Modular Structure**
   - **Issue**: The script lacks functions or classes, making it hard to reuse or test.
   - **Improvement**: Refactor into functions:
     ```python
     def load_data(data_path):
         """Load and return the dataset."""
         try:
             return pd.read_csv(data_path)
         except (FileNotFoundError, pd.errors.ParserError) as e:
             print(f"Error loading data: {e}")
             sys.exit(1)
             
     def create_preprocessor(numeric_cols, categorical_cols):
         """Create and return a column transformer for preprocessing."""
         # Preprocessing logic here
         
     def train_model(X_train, y_train, preprocessor):
         """Train and return a model."""
         # Model training logic here
         
     def evaluate_model(model, X_test, y_test, label_encoder):
         """Evaluate model and print results."""
         # Evaluation logic here
         
     # Main execution
     if __name__ == "__main__":
         data = load_data(raw_data_file)
         # Rest of the workflow
     ```

2. **Add Cross-Validation**
   - **Issue**: Single train-test split may not provide robust evaluation.
   - **Improvement**: Implement cross-validation:
     ```python
     from sklearn.model_selection import cross_val_score
     
     # After defining the model pipeline
     cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
     print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
     ```

3. **Implement Hyperparameter Tuning**
   - **Issue**: Default model parameters may not be optimal.
   - **Improvement**: Add grid search for hyperparameter optimization:
     ```python
     from sklearn.model_selection import GridSearchCV
     
     param_grid = {
         'classifier__C': [0.01, 0.1, 1.0, 10.0],
         'classifier__penalty': ['l1', 'l2'],
         'classifier__solver': ['liblinear', 'saga']
     }
     
     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
     grid_search.fit(X_train, y_train)
     
     print(f"Best parameters: {grid_search.best_params_}")
     print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
     
     # Use the best model for final evaluation
     best_model = grid_search.best_estimator_
     y_pred = best_model.predict(X_test)
     ```

4. **Add Feature Importance Analysis**
   - **Issue**: No insight into which features are most important.
   - **Improvement**: Extract and visualize feature importance:
     ```python
     import matplotlib.pyplot as plt
     import numpy as np
     
     def plot_feature_importance(model, preprocessor, feature_names):
         """Plot feature importance for the trained model."""
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
             plt.savefig('feature_importance.png')
             plt.close()
     
     # After training the model
     plot_feature_importance(model.named_steps['classifier'], 
                            model.named_steps['preprocessor'],
                            numeric_columns + categorical_columns)
     ```

5. **Add Model Persistence**
   - **Issue**: Trained model is not saved for future use.
   - **Improvement**: Save the model using joblib:
     ```python
     import joblib
     
     # After training the model
     model_file = os.path.join(project_root, 'models', 'adult_income_model.joblib')
     os.makedirs(os.path.dirname(model_file), exist_ok=True)
     joblib.dump(model, model_file)
     print(f"Model saved to {model_file}")
     ```

6. **Handle Class Imbalance**
   - **Issue**: No handling of potential class imbalance.
   - **Improvement**: Add class balancing:
     ```python
     from sklearn.utils import class_weight
     
     # After splitting the data
     class_weights = class_weight.compute_class_weight('balanced', 
                                                     classes=np.unique(y_train), 
                                                     y=y_train)
     class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
     
     # Update model definition
     model = Pipeline(steps=[
         ('preprocessor', preprocessor),
         ('classifier', LogisticRegression(class_weight=class_weight_dict))
     ])
     ```

7. **Add Data Exploration**
   - **Issue**: No data exploration before model training.
   - **Improvement**: Add basic exploratory data analysis:
     ```python
     def explore_data(data, target_col):
         """Perform basic exploratory data analysis."""
         print(f"Dataset shape: {data.shape}")
         print("\nMissing values per column:")
         print(data.isnull().sum())
         
         print("\nTarget distribution:")
         target_counts = data[target_col].value_counts(normalize=True)
         print(target_counts)
         
         # For numeric columns
         numeric_cols = data.select_dtypes(include=['number']).columns
         if len(numeric_cols) > 0:
             print("\nNumeric column statistics:")
             print(data[numeric_cols].describe())
         
         return target_counts
     
     # Before preprocessing
     target_distribution = explore_data(data, target)
     ```

8. **Add Command-Line Arguments**
   - **Issue**: Hard-coded parameters limit flexibility.
   - **Improvement**: Add command-line argument parsing:
     ```python
     import argparse
     
     def parse_args():
         """Parse command line arguments."""
         parser = argparse.ArgumentParser(description='Train income prediction model')
         parser.add_argument('--data-path', type=str, help='Path to the dataset')
         parser.add_argument('--test-size', type=float, default=0.2, 
                           help='Proportion of data to use for testing')
         parser.add_argument('--random-state', type=int, default=42, 
                           help='Random seed for reproducibility')
         parser.add_argument('--model-output', type=str, 
                           help='Path to save the trained model')
         return parser.parse_args()
     
     # At the beginning of the script
     if __name__ == "__main__":
         args = parse_args()
         # Use args.data_path, args.test_size, etc.
     ```

9. **Add Logging**
   - **Issue**: No logging for tracking execution.
   - **Improvement**: Implement proper logging:
     ```python
     import logging
     
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
     
     # At the beginning of the script
     logger = setup_logging()
     logger.info("Starting model training pipeline")
     
     # Replace print statements with logger calls
     # e.g., logger.info("Data loaded successfully")
     ```

## Implementation Priority

If implementing all improvements is not feasible, here's a suggested priority order:

1. Fix critical bugs (missing utils.py, data handling, error handling)
2. Add modular structure (refactor into functions)
3. Implement cross-validation and hyperparameter tuning
4. Add model persistence
5. Handle class imbalance
6. Add data exploration and feature importance analysis
7. Implement logging
8. Add command-line arguments