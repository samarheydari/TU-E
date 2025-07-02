"""
Titanic Survival Prediction using Random Forest Classifier

This script processes the Titanic dataset and builds a machine learning model
to predict passenger survival. It includes data preprocessing, feature engineering,
model training, and evaluation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
import warnings

# Configuration constants
RANDOM_STATE = 42
TEST_SIZE = 0.3
CLASS_0_SAMPLE_FRACTION = 1.0  # Changed from 0.6 to avoid artificial imbalance
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        str: Path to the project root directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir

def load_titanic_data(data_path: str = None) -> pd.DataFrame:
    """
    Load Titanic dataset from various possible sources.
    
    Args:
        data_path: Optional path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded Titanic dataset
        
    Raises:
        FileNotFoundError: If no valid dataset is found
    """
    possible_paths = []
    
    if data_path:
        possible_paths.append(data_path)
    
    # Try common locations for Titanic dataset
    project_root = get_project_root()
    possible_paths.extend([
        os.path.join(project_root, "datasets", "titanic", "data.csv"),
        os.path.join(project_root, "data", "titanic.csv"),
        os.path.join(project_root, "titanic.csv"),
        "titanic.csv"
    ])
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Loading data from: {path}")
                return pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to load data from {path}: {e}")
            continue
    
    # If no local file found, try to download from seaborn
    try:
        import seaborn as sns
        logger.info("Loading Titanic dataset from seaborn")
        return sns.load_dataset('titanic')
    except ImportError:
        logger.warning("Seaborn not available for dataset loading")
    except Exception as e:
        logger.warning(f"Failed to load from seaborn: {e}")
    
    # Create sample data if nothing else works
    logger.warning("Creating sample dataset for demonstration")
    return create_sample_titanic_data()

def create_sample_titanic_data() -> pd.DataFrame:
    """
    Create a sample Titanic-like dataset for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample dataset with Titanic-like structure
    """
    np.random.seed(RANDOM_STATE)
    n_samples = 891
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': [f"Passenger_{i}" for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29.7, 14.5, n_samples),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.4, n_samples),
        'Ticket': [f"TICKET_{i}" for i in range(n_samples)],
        'Fare': np.random.lognormal(3.2, 1.0, n_samples),
        'Cabin': [f"C{i}" if np.random.random() > 0.77 else None for i in range(n_samples)],
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    }
    
    # Add some missing values to simulate real data
    age_missing_idx = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
    data['Age'][age_missing_idx] = np.nan
    
    embarked_missing_idx = np.random.choice(n_samples, 2, replace=False)
    for idx in embarked_missing_idx:
        data['Embarked'][idx] = None
    
    return pd.DataFrame(data)

def preprocess_data(data: pd.DataFrame, balance_classes: bool = False) -> pd.DataFrame:
    """
    Preprocess the Titanic dataset.
    
    Args:
        data: Raw Titanic dataset
        balance_classes: Whether to balance classes by sampling
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    logger.info("Starting data preprocessing")
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Standardize column names (seaborn uses different names)
    column_mapping = {
        'pclass': 'Pclass',
        'sex': 'Sex',
        'age': 'Age',
        'sibsp': 'SibSp',
        'parch': 'Parch',
        'fare': 'Fare',
        'embarked': 'Embarked',
        'survived': 'Survived',
        'deck': 'Cabin_Deck',
        'embark_town': 'Embark_Town'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Drop columns that are not useful for prediction
    columns_to_drop = ['who', 'adult_male', 'alive', 'alone', 'Embark_Town', 'class']
    if 'PassengerId' in df.columns:
        columns_to_drop.append('PassengerId')
    
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Handle Cabin/deck information
    if 'Cabin_Deck' in df.columns:
        df['Has_Cabin'] = df['Cabin_Deck'].notna().astype(int)
        # Convert categorical to string to handle missing values
        if df['Cabin_Deck'].dtype.name == 'category':
            df['Cabin_Deck'] = df['Cabin_Deck'].astype(str)
        df['Cabin_Deck'] = df['Cabin_Deck'].fillna('Unknown')
    elif 'Cabin' in df.columns:
        df['Cabin_Deck'] = df['Cabin'].str[0]
        df['Has_Cabin'] = df['Cabin'].notna().astype(int)
        df = df.drop('Cabin', axis=1)
        df['Cabin_Deck'] = df['Cabin_Deck'].fillna('Unknown')
    
    # Handle class imbalance if requested
    if balance_classes and 'Survived' in df.columns:
        logger.info(f"Balancing classes - sampling {CLASS_0_SAMPLE_FRACTION} of class 0")
        df_class_0 = df[df['Survived'] == 0].sample(frac=CLASS_0_SAMPLE_FRACTION, random_state=RANDOM_STATE)
        df_class_1 = df[df['Survived'] == 1]
        df = pd.concat([df_class_0, df_class_1], ignore_index=True)
    
    # Convert categorical columns to strings for easier handling
    categorical_columns = ['Sex', 'Embarked', 'Cabin_Deck']
    for col in categorical_columns:
        if col in df.columns and df[col].dtype.name == 'category':
            df[col] = df[col].astype(str)
    
    # Handle missing values
    logger.info("Handling missing values")
    
    # Age: fill with median
    if 'Age' in df.columns:
        age_median = df['Age'].median()
        df['Age'] = df['Age'].fillna(age_median)
        logger.info(f"Filled missing Age values with median: {age_median:.1f}")
    
    # Embarked: fill with mode
    if 'Embarked' in df.columns:
        embarked_mode = df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'S'
        df['Embarked'] = df['Embarked'].fillna(embarked_mode)
        logger.info(f"Filled missing Embarked values with mode: {embarked_mode}")
    
    # Fare: fill with median (FIXED BUG: was using Age instead of Fare)
    if 'Fare' in df.columns:
        fare_median = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(fare_median)
        logger.info(f"Filled missing Fare values with median: {fare_median:.2f}")
    
    return df

def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using separate LabelEncoders.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    logger.info("Encoding categorical features")
    
    df_encoded = df.copy()
    encoders = {}
    
    categorical_columns = ['Sex', 'Embarked', 'Cabin_Deck']
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            encoders[col] = LabelEncoder()
            df_encoded[col] = encoders[col].fit_transform(df_encoded[col].astype(str))
            logger.info(f"Encoded {col}: {list(encoders[col].classes_)}")
    
    return df_encoded, encoders

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the processed dataset.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    logger.info("Validating processed data")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
        return False
    
    # Check for required columns
    required_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check data types
    numeric_columns = ['Age', 'Fare', 'SibSp', 'Parch']
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"Column {col} should be numeric but is {df[col].dtype}")
            return False
    
    logger.info("Data validation passed")
    return True

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        RandomForestClassifier: Trained model
    """
    logger.info("Training Random Forest model")
    
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Handle any remaining class imbalance
    )
    
    clf.fit(X_train, y_train)
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 5 most important features:")
    for _, row in feature_importance.head().iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return clf

def evaluate_model(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dict: Evaluation metrics
    """
    logger.info("Evaluating model")
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results = {
        'accuracy': accuracy,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred
    }
    
    return results

def main():
    """Main execution function."""
    try:
        logger.info("Starting Titanic survival prediction pipeline")
        
        # Load data
        data = load_titanic_data()
        logger.info(f"Loaded dataset with shape: {data.shape}")
        
        # Preprocess data
        df_processed = preprocess_data(data, balance_classes=False)
        
        # Encode categorical features
        df_encoded, encoders = encode_categorical_features(df_processed)
        
        # Validate data
        if not validate_data(df_encoded):
            raise ValueError("Data validation failed")
        
        print("Missing values after preprocessing:")
        print(df_encoded.isnull().sum())
        
        # Prepare features and target
        X = df_encoded.drop('Survived', axis=1)
        y = df_encoded['Survived']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        print("Missing values in X_train:")
        print(X_train.isnull().sum())
        print("Missing values in X_test:")
        print(X_test.isnull().sum())
        
        # Train model
        clf = train_model(X_train, y_train)
        
        # Evaluate model
        results = evaluate_model(clf, X_test, y_test)
        
        # Print results
        print(f"\nAccuracy: {results['accuracy']:.4f}")
        print(f"\nClassification Report:\n{results['classification_report']}")
        print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
