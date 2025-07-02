import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for importing utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Try to import get_project_root, with fallback if not available
try:
    from utils import get_project_root
    project_root = get_project_root()
except ImportError:
    logger.warning("utils module not found. Using current directory as project root.")
    project_root = current_dir

# Configuration parameters
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'target_column': 'score_text',
    'data_path': os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv"),
    'model_params': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'cv_folds': 5,
    'use_class_weights': True,
    'use_hyperparameter_tuning': True,
    'output_dir': os.path.join(project_root, "output")
}

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

def load_data(file_path):
    """Load and validate the dataset."""
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("Dataset is empty")
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Dataset not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def explore_data(data, target_col):
    """Perform exploratory data analysis."""
    logger.info("Performing exploratory data analysis")
    
    # Basic statistics
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data types:\n{data.dtypes}")
    logger.info(f"Missing values:\n{data.isna().sum()}")
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    target_counts = data[target_col].value_counts()
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title(f'Distribution of {target_col}')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'target_distribution.png'))
    logger.info(f"Target distribution saved to {os.path.join(CONFIG['output_dir'], 'target_distribution.png')}")
    
    # Check for class imbalance
    logger.info(f"Class distribution:\n{target_counts}")
    imbalance_ratio = target_counts.max() / target_counts.min()
    logger.info(f"Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    return target_counts

def preprocess_data(data, target_col):
    """Preprocess the data and split into features and target."""
    if target_col not in data.columns:
        logger.error(f"Target column '{target_col}' not found in dataset")
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    logger.info("Preprocessing data")
    
    # Split features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Encode target variable
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    logger.info(f"Target classes: {le_y.classes_}")
    
    # Store original column names by type for later use
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    logger.info(f"Found {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
    
    return X, y_encoded, le_y, numerical_cols, categorical_cols

def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Create a preprocessing pipeline with appropriate imputation strategies."""
    logger.info("Creating preprocessing pipeline")
    
    # Create preprocessing pipelines for each column type
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def train_and_evaluate_model(X, y, numerical_cols, categorical_cols, le_y, test_size=0.2, random_state=42):
    """Split data, train model, and evaluate."""
    logger.info("Splitting data into train and test sets")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    # Compute class weights if enabled
    if CONFIG['use_class_weights']:
        logger.info("Computing class weights")
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        logger.info(f"Class weights: {class_weight_dict}")
    else:
        class_weight_dict = None
    
    # Create base classifier
    base_clf = RandomForestClassifier(
        random_state=random_state,
        class_weight=class_weight_dict
    )
    
    # Create a preprocessing and modeling pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_clf)
    ])
    
    # Perform hyperparameter tuning if enabled
    if CONFIG['use_hyperparameter_tuning']:
        logger.info("Performing hyperparameter tuning")
        param_grid = {
            'classifier__' + key: value for key, value in CONFIG['model_params'].items()
        }
        
        grid_search = GridSearchCV(
            model_pipeline,
            param_grid=param_grid,
            cv=CONFIG['cv_folds'],
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        logger.info("Fitting grid search")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Use the best model
        best_model = grid_search.best_estimator_
    else:
        logger.info("Training model without hyperparameter tuning")
        model_pipeline.fit(X_train, y_train)
        best_model = model_pipeline
    
    # Perform cross-validation
    logger.info("Performing cross-validation")
    cv_scores = cross_val_score(
        best_model, X_train, y_train, 
        cv=CONFIG['cv_folds'], scoring='accuracy'
    )
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Make predictions
    logger.info("Making predictions on test set")
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{report}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_y.classes_, yticklabels=le_y.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'))
    
    # Try to get feature importances if available
    try:
        if hasattr(best_model[-1], 'feature_importances_'):
            feature_importances = best_model[-1].feature_importances_
            
            # Get feature names from the preprocessor
            feature_names = []
            for name, trans, cols in best_model[0].transformers_:
                if name == 'cat' and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = trans.named_steps['onehot'].get_feature_names_out(cols)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend(cols)
            
            # Plot top 20 features
            indices = np.argsort(feature_importances)[::-1][:20]
            plt.figure(figsize=(12, 8))
            plt.title("Top 20 Feature Importances")
            plt.bar(range(len(indices)), feature_importances[indices], align="center")
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG['output_dir'], 'feature_importance.png'))
            logger.info(f"Feature importance plot saved to {os.path.join(CONFIG['output_dir'], 'feature_importance.png')}")
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
    
    return best_model, accuracy, report, y_test, y_pred

def save_model(model, path):
    """Save the trained model to disk."""
    try:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def main():
    """Main function to run the pipeline."""
    logger.info("Starting the machine learning pipeline")
    
    # Load data
    raw_data = load_data(CONFIG['data_path'])
    
    # Explore data
    target_counts = explore_data(raw_data, CONFIG['target_column'])
    
    # Preprocess data
    X, y, le_y, numerical_cols, categorical_cols = preprocess_data(
        raw_data, CONFIG['target_column']
    )
    
    # Train and evaluate model
    model, accuracy, report, y_test, y_pred = train_and_evaluate_model(
        X, y, numerical_cols, categorical_cols, le_y,
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state']
    )
    
    # Save the model
    model_path = os.path.join(CONFIG['output_dir'], "random_forest_model.joblib")
    save_model(model, model_path)
    
    logger.info("Machine learning pipeline completed successfully")

if __name__ == "__main__":
    main()