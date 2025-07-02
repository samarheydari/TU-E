import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Path configuration with error handling
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    from utils import get_project_root
    project_root = get_project_root()
except ImportError:
    print("Warning: utils module not found. Using current directory as project root.")
    project_root = os.path.dirname(os.path.abspath(__file__))

# Define functions for modular code structure
def load_data(file_path):
    """Load and validate the dataset."""
    try:
        print(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        
        # Validate required columns
        if 'Diabetes_binary' not in data.columns:
            raise ValueError("Required column 'Diabetes_binary' not found in dataset")
            
        print(f"Dataset loaded successfully with shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_data(data):
    """Analyze dataset characteristics and distributions."""
    print("\n=== Data Analysis ===")
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    # Class distribution
    class_counts = data['Diabetes_binary'].value_counts()
    print(f"Class distribution:\n{class_counts}")
    print(f"Class distribution (%):\n{class_counts / len(data) * 100}")
    
    # Check for potential class imbalance
    if len(class_counts) > 1 and class_counts.min() / class_counts.max() < 0.2:
        print("WARNING: Significant class imbalance detected")
    
    return class_counts

def analyze_correlations(data):
    """Analyze feature correlations to identify potential spurious relationships."""
    print("\n=== Correlation Analysis ===")
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Display correlations with target variable
    target_corrs = corr_matrix['Diabetes_binary'].sort_values(ascending=False)
    print("Correlations with target variable:")
    print(target_corrs)
    
    # Identify potential multicollinearity
    high_corr_features = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8 and i != j:
                high_corr_features.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    
    if high_corr_features:
        print("\nPotential multicollinearity detected between:")
        for feat1, feat2 in high_corr_features:
            print(f"  - {feat1} and {feat2}")
    
    # Optional: Save correlation heatmap
    try:
        import seaborn as sns
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, 'correlation_matrix.png'))
        plt.close()
        print("Correlation matrix visualization saved to correlation_matrix.png")
    except ImportError:
        print("Seaborn not installed. Skipping correlation visualization.")
    
    return high_corr_features

def select_features(X, y):
    """Select features using multiple methods to reduce spurious correlations."""
    print("\n=== Feature Selection ===")
    
    # Method 1: Statistical significance (ANOVA F-value)
    selector1 = SelectKBest(f_classif, k=min(10, X.shape[1]))
    selector1.fit(X, y)
    selected_features1 = X.columns[selector1.get_support()]
    
    # Method 2: Feature importance from Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Display feature importances
    print("Feature importances:")
    print(feature_importances.head(10))
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['feature'][:10], feature_importances['importance'][:10])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'feature_importance.png'))
    plt.close()
    
    # Return top features based on importance
    top_features = feature_importances['feature'][:min(10, len(feature_importances))].tolist()
    print(f"Selected top {len(top_features)} features based on importance")
    
    return top_features

def handle_class_imbalance(X_train, y_train, class_counts):
    """Handle class imbalance if present."""
    if len(class_counts) > 1 and class_counts.min() / class_counts.max() < 0.2:
        print("\n=== Handling Class Imbalance ===")
        
        # Option 1: Class weights
        class_weights = {0: 1, 1: class_counts[0] / class_counts[1]} if 1 in class_counts else {0: 1}
        print(f"Using class weights: {class_weights}")
        
        return class_weights
    
    return None

def train_model(X_train, y_train, class_weights=None):
    """Train the model with given parameters."""
    print("\n=== Model Training ===")
    
    # Create model with appropriate parameters
    params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if class_weights:
        params['class_weight'] = class_weights
    
    model = RandomForestClassifier(**params)
    
    # Train model
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    print("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test, X, y):
    """Evaluate the model with comprehensive metrics."""
    print("\n=== Model Evaluation ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else None
    
    # Basic metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # ROC-AUC (only for binary classification)
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC-AUC score: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {e}")
    
    # Cross-validation for more robust evaluation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"Could not perform cross-validation: {e}")

def save_model(model, model_path):
    """Save the trained model to disk."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Main execution flow
if __name__ == "__main__":
    # Define file paths
    raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "diabetes_rf_model.pkl")
    
    # Load and analyze data
    data = load_data(raw_data_file)
    class_counts = analyze_data(data)
    analyze_correlations(data)
    
    # Prepare features and target
    X = data.drop(columns=['Diabetes_binary'])
    y = data['Diabetes_binary']
    
    # Feature selection
    selected_features = select_features(X, y)
    X_selected = X[selected_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Handle class imbalance
    class_weights = handle_class_imbalance(X_train, y_train, class_counts)
    
    # Train model
    model = train_model(X_train, y_train, class_weights)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, X_selected, y)
    
    # Save model
    save_model(model, model_path)
    
    print("\nDiabetes prediction model pipeline completed successfully!")
