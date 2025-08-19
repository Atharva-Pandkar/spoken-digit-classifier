import numpy as np
import time
from pathlib import Path
import hashlib
import json
import joblib
from typing import Dict, Any

from data.dataset import FSIDDataset
from features.audio_io import AudioProcessor
from features.mfcc import MFCCExtractor
from features.mel_spectrogram import MelSpectrogramExtractor
from models.classical import ClassicalModel
from models.cnn_model import CNNModel
from utils.timing import Timer
from utils.io import save_artifacts

def get_cache_key(config):
    """Generate cache key from config"""
    key_parts = [
        config.get('model_type', 'SVM'),
        str(config.get('test_size', 0.2)),
        str(config.get('mfcc_coeffs', 20)),
        str(config.get('use_deltas', True)),
        str(config.get('epochs', 50)),
        str(config.get('random_seed', 42))
    ]
    return '_'.join(key_parts).replace(' ', '_').replace('(', '').replace(')', '')

def save_cached_model(artifacts, config):
    """Save model artifacts to cache"""
    try:
        cache_dir = Path('model_cache')
        cache_dir.mkdir(exist_ok=True)
        
        cache_key = get_cache_key(config)
        cache_path = cache_dir / f'{cache_key}_artifacts.joblib'
        
        print(f"💾 Caching model: {cache_path}")
        
        # Create a copy without the large objects for caching
        cache_data = artifacts.copy()
        
        # Save model separately
        if config.get('model_type') == 'CNN':
            model_path = cache_dir / f'{cache_key}_cnn_model.keras'
            artifacts['model'].save_model(str(model_path))
            cache_data['model_path'] = str(model_path)
            cache_data['model'] = None  # Don't store model directly
        else:
            model_path = cache_dir / f'{cache_key}_model.joblib'
            joblib.dump(artifacts['model'], model_path)
            cache_data['model_path'] = str(model_path)
            cache_data['model'] = None  # Don't store model directly
        
        # Save feature extractor separately
        scaler_path = cache_dir / f'{cache_key}_scaler.joblib'
        artifacts['feature_extractor'].save_scaler(scaler_path)
        cache_data['scaler_path'] = str(scaler_path)
        cache_data['feature_extractor'] = None  # Don't store extractor directly
        
        joblib.dump(cache_data, cache_path)
        print(f"✅ Model cached successfully!")
        
    except Exception as e:
        print(f"⚠️ Caching failed: {e}")

def load_cached_model(config):
    """Load model artifacts from cache"""
    try:
        cache_dir = Path('model_cache')
        if not cache_dir.exists():
            return None
            
        cache_key = get_cache_key(config)
        cache_path = cache_dir / f'{cache_key}_artifacts.joblib'
        
        if not cache_path.exists():
            return None
            
        print(f"📦 Loading cached model: {cache_path}")
        
        # Load cached data
        cache_data = joblib.load(cache_path)
        
        # Reconstruct model
        if config.get('model_type') == 'CNN':
            model = CNNModel()
            model.load_model(cache_data['model_path'])
        else:
            model = joblib.load(cache_data['model_path'])
        
        # Reconstruct feature extractor
        if config.get('model_type') == 'CNN':
            feature_extractor = MelSpectrogramExtractor()
        else:
            feature_extractor = MFCCExtractor(
                n_mfcc=config.get('mfcc_coeffs', 20),
                use_delta=config.get('use_deltas', True),
                use_delta_deltas=config.get('use_delta_deltas', True)
            )
        feature_extractor.load_scaler(cache_data['scaler_path'])
        
        # Restore full artifacts
        cache_data['model'] = model
        cache_data['feature_extractor'] = feature_extractor
        
        print(f"✅ Cached model loaded! Accuracy: {cache_data.get('test_accuracy', 0):.3f}")
        return cache_data
        
    except Exception as e:
        print(f"⚠️ Cache loading failed: {e}")
        return None

def train_model(dataset: FSIDDataset, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train a spoken digit classifier with caching"""
    
    # Check for cached model first
    cached_result = load_cached_model(config)
    if cached_result is not None:
        return cached_result
    
    print("Starting model training...")
    
    # Extract configuration
    model_type = config.get('model_type', 'SVM (RBF)')
    test_size = config.get('test_size', 0.2)
    val_size = config.get('val_size', 0.15)
    random_seed = config.get('random_seed', 42)
    
    # Feature extraction parameters
    mfcc_coeffs = config.get('mfcc_coeffs', 20)
    use_deltas = config.get('use_deltas', True)
    use_delta_deltas = config.get('use_delta_deltas', True)
    
    # CNN parameters - improved for better accuracy
    epochs = config.get('epochs', 100)  # Increased epochs
    batch_size = config.get('batch_size', 64)  # Larger batch size
    
    # Determine if we're using CNN
    is_cnn = model_type == 'CNN'
    
    # Create data splits
    print("Creating speaker-holdout data splits...")
    splits = dataset.create_speaker_holdout_split(
        test_size=test_size,
        val_size=val_size,
        random_state=random_seed
    )
    
    # Initialize components
    audio_processor = AudioProcessor(target_sr=8000)
    
    if is_cnn:
        feature_extractor = MelSpectrogramExtractor(
            n_mels=60,
            target_length=80
        )
    else:
        feature_extractor = MFCCExtractor(
            n_mfcc=mfcc_coeffs,
            use_deltas=use_deltas,
            use_delta_deltas=use_delta_deltas
        )
    
    # Extract features for each split
    print("Extracting features...")
    
    def extract_features_for_split(split_data):
        """Extract features for a data split"""
        features = []
        labels = []
        error_count = 0
        
        for _, row in split_data.iterrows():
            try:
                # Process audio
                audio = audio_processor.preprocess_audio(row['filepath'])
                
                # Extract features
                feature_vector = feature_extractor.extract_features(audio, 8000)
                
                features.append(feature_vector)
                labels.append(row['digit'])
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only print first 5 errors
                    print(f"Error processing {row['filepath']}: {str(e)[:100]}...")
                elif error_count == 6:
                    print("... (suppressing further error messages)")
                continue
        
        if error_count > 0:
            print(f"Skipped {error_count} files due to processing errors")
        
        if len(features) == 0:
            raise ValueError("No valid features extracted from any files")
        
        return np.array(features), np.array(labels)
    
    # Extract features for all splits
    with Timer() as feature_timer:
        X_train, y_train = extract_features_for_split(splits['train'])
        X_val, y_val = extract_features_for_split(splits['val'])
        X_test, y_test = extract_features_for_split(splits['test'])
    
    print(f"Feature extraction completed in {feature_timer.get_elapsed_ms():.2f} ms")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Fit feature scaler and prepare data
    print("Fitting feature scaler...")
    X_train_scaled = feature_extractor.fit_transform_features(X_train)
    X_val_scaled = feature_extractor.transform_features(X_val)
    X_test_scaled = feature_extractor.transform_features(X_test)
    
    # Initialize and train model
    print(f"Training {model_type} model...")
    
    if is_cnn:
        model = CNNModel()
        
        with Timer() as training_timer:
            training_info = model.train(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                epochs=epochs,
                batch_size=batch_size
            )
    else:
        model = ClassicalModel(model_type=model_type)
        
        with Timer() as training_timer:
            training_info = model.train(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                use_grid_search=True
            )
    
    training_time = training_timer.get_elapsed_ms() / 1000  # Convert to seconds
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(X_test_scaled, y_test)
    
    # Benchmark inference time
    print("Benchmarking inference time...")
    inference_times = []
    
    for i in range(min(20, len(X_test_scaled))):  # Test on up to 20 samples
        sample = X_test_scaled[i:i+1]
        
        start_time = time.time()
        _ = model.predict(sample)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # milliseconds
        inference_times.append(inference_time)
    
    avg_inference_time = np.mean(inference_times)
    p95_inference_time = np.percentile(inference_times, 95)
    
    # Create artifacts dictionary
    artifacts = {
        # Models and scalers
        'model': model,
        'scaler': feature_extractor.scaler,
        'feature_extractor': feature_extractor,
        'is_cnn': is_cnn,
        
        # Configuration
        'feature_config': feature_extractor.get_config(),
        'training_config': config,
        
        # Performance metrics
        'train_accuracy': training_info.get('train_accuracy', 0.0),
        'val_accuracy': training_info.get('val_accuracy', 0.0),
        'test_accuracy': test_results['accuracy'],
        
        # Detailed evaluation
        'confusion_matrix': test_results['confusion_matrix'],
        'classification_report': test_results['classification_report'],
        'per_digit_accuracy': test_results['per_class_accuracy'],
        
        # Timing information
        'training_time': training_time,
        'avg_inference_time_ms': avg_inference_time,
        'p95_inference_time_ms': p95_inference_time,
        'latency_distribution': inference_times,
        
        # Model information
        'model_type': model_type,
        'feature_dimension': feature_extractor.get_feature_dimension(),
        
        # Additional training info
        'best_params': training_info.get('best_params', {}),
        'best_cv_score': training_info.get('best_cv_score', 0.0)
    }
    
    # Calculate model size
    if is_cnn:
        model_size_mb = model.get_model_size()
    else:
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp_file:
            model.save_model(tmp_file.name)
            model_size_mb = model.get_model_size(tmp_file.name)
    
    artifacts['model_size_mb'] = model_size_mb
    
    print("\n=== Training Results ===")
    print(f"Training Accuracy: {artifacts['train_accuracy']:.3f}")
    print(f"Validation Accuracy: {artifacts['val_accuracy']:.3f}")
    print(f"Test Accuracy: {artifacts['test_accuracy']:.3f}")
    print(f"Average Inference Time: {avg_inference_time:.1f} ms")
    print(f"95th Percentile Inference Time: {p95_inference_time:.1f} ms")
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    # Check targets
    accuracy_target = artifacts['test_accuracy'] >= 0.95
    latency_target = avg_inference_time <= 200
    
    print(f"\nTarget Achievement:")
    print(f"Accuracy ≥95%: {'✅' if accuracy_target else '❌'}")
    print(f"Latency ≤200ms: {'✅' if latency_target else '❌'}")
    
    # Cache the trained model for future use
    save_cached_model(artifacts, config)
    
    return artifacts

if __name__ == "__main__":
    # Example usage
    dataset = FSIDDataset()
    dataset.load_dataset()
    
    config = {
        'model_type': 'SVM (RBF)',
        'test_size': 0.2,
        'val_size': 0.15,
        'mfcc_coeffs': 20,
        'use_deltas': True,
        'use_delta_deltas': True,
        'random_seed': 42
    }
    
    artifacts = train_model(dataset, config)
    
    # Save artifacts
    save_artifacts(artifacts, "./model_artifacts")
