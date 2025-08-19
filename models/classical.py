import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os
from typing import Dict, Tuple, Any, Optional

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class ClassicalModel:
    """Wrapper for classical ML models for digit classification"""
    
    def __init__(self, model_type: str = "SVM (RBF)"):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on type"""
        if self.model_type == "SVM (RBF)":
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
            self.param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.01, 0.1]
            }
        
        elif self.model_type == "SVM (Linear)":
            self.model = SVC(kernel='linear', probability=True, random_state=42)
            self.param_grid = {
                'C': [0.1, 1, 10]
            }
        
        elif self.model_type == "Logistic Regression":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.param_grid = {
                'C': [0.5, 1, 2],
                'solver': ['liblinear', 'lbfgs']
            }
        
        elif self.model_type == "XGBoost":
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    random_state=42,
                    n_estimators=100
                )
                self.param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.1, 0.2],
                    'n_estimators': [50, 100]
                }
            else:
                # Fallback to Logistic Regression
                print("XGBoost not available, falling back to Logistic Regression")
                self.model_type = "Logistic Regression"
                self._initialize_model()
                return
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              use_grid_search: bool = True) -> Dict[str, Any]:
        """Train the model with optional hyperparameter tuning"""
        
        if use_grid_search and len(self.param_grid) > 0:
            # Use grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                self.model, 
                self.param_grid, 
                cv=3, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            training_info = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_
            }
        else:
            # Simple training without grid search
            self.model.fit(X_train, y_train)
            training_info = {'method': 'simple_training'}
        
        self.is_trained = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        training_info['train_accuracy'] = train_accuracy
        
        # Calculate validation accuracy if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            training_info['val_accuracy'] = val_accuracy
        
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Per-class accuracy
        per_class_acc = {}
        for i in range(10):  # Digits 0-9
            mask = y_test == i
            if np.sum(mask) > 0:
                per_class_acc[i] = accuracy_score(y_test[mask], y_pred[mask])
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'per_class_accuracy': per_class_acc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
    
    def get_model_size(self, filepath: str) -> float:
        """Get model size in MB"""
        if os.path.exists(filepath):
            size_bytes = os.path.getsize(filepath)
            return size_bytes / (1024 * 1024)  # Convert to MB
        return 0.0
