import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging

# Configuration
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'k_features': 10,
    'max_iter': 1000,
    'cv_folds': 5
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def load_data(file_path):
    """Load data with error handling."""
    try:
        if not file_path.exists():
            logger.warning(f"Dataset not found at {file_path}")
            # Create sample data for demonstration
            logger.info("Creating sample diabetes dataset for demonstration...")
            return create_sample_diabetes_data()
        
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Creating sample diabetes dataset for demonstration...")
        return create_sample_diabetes_data()

def create_sample_diabetes_data():
    """Create a sample diabetes dataset for demonstration."""
    np.random.seed(CONFIG['random_state'])
    n_samples = 1000
    n_features = 15
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some correlation to features
    y_prob = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1)))
    y = (y_prob > 0.5).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['Diabetes_binary'] = y
    
    logger.info(f"Created sample dataset with {n_samples} samples and {n_features} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return data

project_root = get_project_root()
raw_data_file = project_root / "datasets" / "diabetes_indicator" / "5050_split.csv"

# Create datasets directory if it doesn't exist
raw_data_file.parent.mkdir(parents=True, exist_ok=True)

data = load_data(raw_data_file)

def preprocess_data(X_train, X_test, y_train):
    """Improved preprocessing pipeline with proper order."""
    logger.info("Starting data preprocessing...")
    
    # Step 1: Scale the data first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Apply SMOTE on scaled data
    smote = SMOTE(random_state=CONFIG['random_state'])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    logger.info(f"After SMOTE - Training samples: {X_train_resampled.shape[0]}")
    logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    
    # Step 3: Feature selection on resampled data
    selector = SelectKBest(f_classif, k=min(CONFIG['k_features'], X_train_resampled.shape[1]))
    X_train_final = selector.fit_transform(X_train_resampled, y_train_resampled)
    X_test_final = selector.transform(X_test_scaled)
    
    selected_features = selector.get_support(indices=True)
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    
    return X_train_final, X_test_final, y_train_resampled, scaler, selector

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train model with cross-validation and comprehensive evaluation."""
    logger.info("Training logistic regression model...")
    
    model = LogisticRegression(
        max_iter=CONFIG['max_iter'],
        random_state=CONFIG['random_state']
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=CONFIG['cv_folds'], scoring='roc_auc')
    logger.info(f"Cross-validation ROC-AUC scores: {cv_scores}")
    logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Comprehensive evaluation
    logger.info("\n" + "="*50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
    except ValueError as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    return model

def save_model_and_preprocessors(model, scaler, selector, model_dir="models"):
    """Save trained model and preprocessors."""
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    try:
        joblib.dump(model, model_path / "diabetes_model.pkl")
        joblib.dump(scaler, model_path / "scaler.pkl")
        joblib.dump(selector, model_path / "feature_selector.pkl")
        logger.info(f"Model and preprocessors saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def main():
    """Main execution function."""
    logger.info("Starting diabetes prediction pipeline...")
    
    # Data preparation
    X = data.drop('Diabetes_binary', axis=1)
    y = data['Diabetes_binary']
    
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    # Preprocessing
    X_train_processed, X_test_processed, y_train_processed, scaler, selector = preprocess_data(
        X_train, X_test, y_train
    )
    
    # Model training and evaluation
    model = train_and_evaluate_model(
        X_train_processed, X_test_processed, y_train_processed, y_test
    )
    
    # Save model
    save_model_and_preprocessors(model, scaler, selector)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
