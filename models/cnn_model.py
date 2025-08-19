import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any, Tuple
import tempfile
import os
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

class CNNModel:
    """CNN model for spoken digit classification with mel-spectrogram input"""
    
    def __init__(self):
        self.model = None
        self.input_shape = (60, 80, 1)  # Height, Width, Channels for mel-spectrogram
        self.num_classes = 10
        self.is_trained = False
        
    def build_model(self):
        """Build the CNN model with the specified architecture"""
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # Enhanced architecture for better accuracy
        # First Conv Block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # More filters
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second Conv Block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # More filters
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Third Conv Block  
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)  # More filters
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Fourth Conv Block for better feature extraction
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Enhanced Dense layers
        x = layers.Dense(512, activation='relu')(x)  # Larger dense layer
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs, name='functional')
        
        # Compile model with better optimizer
        from tensorflow.keras.optimizers import Adam
        self.model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train the CNN model"""
        
        if self.model is None:
            self.build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Enhanced callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if validation_data else 'accuracy',
                patience=15,  # More patience
                restore_best_weights=True,
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,  # Less aggressive reduction
                patience=8,  # More patience
                min_lr=1e-7,
                min_delta=0.001
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='best_model_weights.keras',
                monitor='val_accuracy' if validation_data else 'accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=0
            )
        ]
        
        # Add Streamlit progress callback if available
        if STREAMLIT_AVAILABLE:
            callbacks.append(StreamlitProgressCallback(epochs))
        
        # Train model
        verbose = 0 if STREAMLIT_AVAILABLE else 1
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Calculate final accuracies
        train_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0.0
        
        training_info = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'history': history.history,
            'epochs_trained': len(history.history['accuracy'])
        }
        
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure correct input shape
        if len(X.shape) == 3:
            X = np.expand_dims(X, -1)  # Add channel dimension
        
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure correct input shape
        if len(X.shape) == 3:
            X = np.expand_dims(X, -1)  # Add channel dimension
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure correct input shape
        if len(X_test.shape) == 3:
            X_test = np.expand_dims(X_test, -1)  # Add channel dimension
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
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
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
    
    def get_model_size(self, filepath: str = None) -> float:
        """Get model size in MB"""
        if filepath and os.path.exists(filepath):
            # If filepath is provided, get size of saved model
            size_bytes = 0
            if os.path.isdir(filepath):
                for dirpath, dirnames, filenames in os.walk(filepath):
                    for filename in filenames:
                        size_bytes += os.path.getsize(os.path.join(dirpath, filename))
            else:
                size_bytes = os.path.getsize(filepath)
            return size_bytes / (1024 * 1024)  # Convert to MB
        elif self.model is not None:
            # Estimate size based on parameters
            total_params = self.model.count_params()
            # Rough estimate: 4 bytes per parameter (float32)
            size_bytes = total_params * 4
            return size_bytes / (1024 * 1024)  # Convert to MB
        return 0.0
    
    def summary(self):
        """Print model summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")

class StreamlitProgressCallback(keras.callbacks.Callback):
    """Custom callback to show training progress in Streamlit"""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = None
        self.status_text = None
        self.metrics_placeholder = None
        
    def on_train_begin(self, logs=None):
        if STREAMLIT_AVAILABLE:
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.metrics_placeholder = st.empty()
    
    def on_epoch_end(self, epoch, logs=None):
        if STREAMLIT_AVAILABLE and self.progress_bar is not None:
            progress = (epoch + 1) / self.total_epochs
            self.progress_bar.progress(progress)
            
            # Update status
            self.status_text.text(f'Epoch {epoch + 1}/{self.total_epochs}')
            
            # Update metrics
            if logs:
                metrics_text = f"Loss: {logs.get('loss', 0):.4f}"
                if 'accuracy' in logs:
                    metrics_text += f" - Accuracy: {logs['accuracy']:.4f}"
                if 'val_loss' in logs:
                    metrics_text += f" - Val Loss: {logs['val_loss']:.4f}"
                if 'val_accuracy' in logs:
                    metrics_text += f" - Val Accuracy: {logs['val_accuracy']:.4f}"
                
                self.metrics_placeholder.text(metrics_text)
    
    def on_train_end(self, logs=None):
        if STREAMLIT_AVAILABLE and self.progress_bar is not None:
            self.progress_bar.progress(1.0)
            self.status_text.text('Training completed!')