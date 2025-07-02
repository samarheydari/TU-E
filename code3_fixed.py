import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path for importing utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Try to import get_project_root, with fallback if not available
try:
    from utils import get_project_root
    project_root = get_project_root()
except ImportError:
    print("Warning: utils module not found. Using current directory as project root.")
    project_root = current_dir

# Configuration parameters
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'target_column': 'score_text',
    'data_path': os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
}

def load_data(file_path):
    """Load and validate the dataset."""
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("Dataset is empty")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data, target_col):
    """Preprocess the data and split into features and target."""
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Split features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Encode target variable
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    # Store original column names by type for later use
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Encode categorical features in X
    label_encoders = {}
    for column in categorical_cols:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    
    return X, y_encoded, label_encoders, numerical_cols, categorical_cols

def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Create a preprocessing pipeline with appropriate imputation strategies."""
    # Create preprocessing pipelines for each column type
    numerical_transformer = Pipeline(steps=[
        # For numerical features, use median imputation
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        # For categorical features, use most frequent value imputation
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def train_and_evaluate_model(X, y, numerical_cols, categorical_cols, test_size=0.2, random_state=42):
    """Split data, train model, and evaluate."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    # Create a preprocessing and modeling pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model_pipeline, accuracy, report, y_test, y_pred

def main():
    """Main function to run the pipeline."""
    # Load data
    raw_data = load_data(CONFIG['data_path'])
    print(f"Loaded data with shape: {raw_data.shape}")
    
    # Preprocess data
    X, y, label_encoders, numerical_cols, categorical_cols = preprocess_data(
        raw_data, CONFIG['target_column']
    )
    print(f"Preprocessed data: {len(numerical_cols)} numerical features, {len(categorical_cols)} categorical features")
    
    # Check for missing values
    missing_values = X.isna().sum().sum()
    print(f"Total missing values in features: {missing_values}")
    
    # Train and evaluate model
    model, accuracy, report, y_test, y_pred = train_and_evaluate_model(
        X, y, numerical_cols, categorical_cols, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state']
    )
    
    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Classification report:\n{report}")
    
    # Optional: Save the model
    # import joblib
    # model_path = os.path.join(project_root, "models", "random_forest_model.joblib")
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # joblib.dump(model, model_path)
    # print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()