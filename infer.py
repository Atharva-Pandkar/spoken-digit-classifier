import numpy as np
import time
from pathlib import Path
from typing import Tuple, Dict, Any

from features.audio_io import AudioProcessor
from features.mfcc import MFCCExtractor
from features.mel_spectrogram import MelSpectrogramExtractor
from models.classical import ClassicalModel
from models.cnn_model import CNNModel
from utils.timing import Timer

def predict_digit(audio_filepath: str, artifacts: Dict[str, Any]) -> Tuple[int, float]:
    """Predict digit from audio file"""
    
    # Extract components from artifacts
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_config = artifacts['feature_config']
    is_cnn = artifacts.get('is_cnn', False)
    
    # Initialize components
    audio_processor = AudioProcessor(target_sr=8000)
    
    if is_cnn:
        feature_extractor = MelSpectrogramExtractor(
            n_mels=feature_config.get('n_mels', 60),
            target_length=feature_config.get('target_length', 80)
        )
    else:
        feature_extractor = MFCCExtractor(
            n_mfcc=feature_config['n_mfcc'],
            use_deltas=feature_config['use_deltas'],
            use_delta_deltas=feature_config['use_delta_deltas']
        )
    
    feature_extractor.scaler = scaler
    feature_extractor.is_fitted = True
    
    # Process audio and extract features
    with Timer() as processing_timer:
        # Load and preprocess audio
        audio = audio_processor.preprocess_audio(audio_filepath)
        
        # Extract features
        features = feature_extractor.extract_features(audio, 8000)
        
        if is_cnn:
            # For CNN, features are already in the right shape (H, W, C)
            features_scaled = feature_extractor.transform_features(features.reshape(1, *features.shape))
        else:
            # For classical models, flatten the features
            features_scaled = feature_extractor.transform_features(features.reshape(1, -1))
    
    # Make prediction
    with Timer() as inference_timer:
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
    
    total_time = processing_timer.get_elapsed_ms() + inference_timer.get_elapsed_ms()
    
    return prediction, confidence

def batch_predict(audio_files: list, artifacts: Dict[str, Any]) -> list:
    """Perform batch prediction on multiple audio files"""
    
    results = []
    
    for filepath in audio_files:
        try:
            start_time = time.time()
            prediction, confidence = predict_digit(filepath, artifacts)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # milliseconds
            
            results.append({
                'filepath': filepath,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'inference_time_ms': inference_time
            })
            
        except Exception as e:
            results.append({
                'filepath': filepath,
                'prediction': None,
                'confidence': 0.0,
                'inference_time_ms': 0.0,
                'error': str(e)
            })
    
    return results

def predict_from_audio_array(audio: np.ndarray, sample_rate: int, artifacts: Dict[str, Any]) -> Tuple[int, float]:
    """Predict digit from audio array (for microphone input)"""
    
    # Extract components from artifacts
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_config = artifacts['feature_config']
    is_cnn = artifacts.get('is_cnn', False)
    
    # Initialize components
    audio_processor = AudioProcessor(target_sr=8000)
    
    if is_cnn:
        feature_extractor = MelSpectrogramExtractor(
            n_mels=feature_config.get('n_mels', 60),
            target_length=feature_config.get('target_length', 80)
        )
    else:
        feature_extractor = MFCCExtractor(
            n_mfcc=feature_config['n_mfcc'],
            use_deltas=feature_config['use_deltas'],
            use_delta_deltas=feature_config['use_delta_deltas']
        )
    
    feature_extractor.scaler = scaler
    feature_extractor.is_fitted = True
    
    # Resample if necessary
    if sample_rate != 8000:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * 8000 / sample_rate))
    
    # Normalize and trim
    audio = audio_processor.normalize_audio(audio)
    audio = audio_processor.trim_silence(audio)
    
    # Ensure minimum length
    min_length = int(0.1 * 8000)  # 100ms minimum
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
    
    # Extract features
    features = feature_extractor.extract_features(audio, 8000)
    
    if is_cnn:
        # For CNN, features are already in the right shape
        features_scaled = feature_extractor.transform_features(features.reshape(1, *features.shape))
    else:
        # For classical models, flatten the features
        features_scaled = feature_extractor.transform_features(features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = np.max(probabilities)
    
    return prediction, confidence

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from utils.io import load_artifacts
    
    parser = argparse.ArgumentParser(description="Spoken Digit Inference")
    parser.add_argument("--file", type=str, help="Single audio file for inference")
    parser.add_argument("--dir", type=str, help="Directory of audio files")
    parser.add_argument("--artifacts", type=str, default="./model_artifacts", 
                       help="Path to model artifacts directory")
    parser.add_argument("--out", type=str, help="Output CSV file for batch results")
    
    args = parser.parse_args()
    
    # Load artifacts
    print("Loading model artifacts...")
    artifacts = load_artifacts(args.artifacts)
    
    if args.file:
        # Single file inference
        print(f"Processing file: {args.file}")
        prediction, confidence = predict_digit(args.file, artifacts)
        print(f"Predicted digit: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        
    elif args.dir:
        # Batch inference
        print(f"Processing directory: {args.dir}")
        
        # Find all WAV files
        audio_files = list(Path(args.dir).glob("*.wav"))
        print(f"Found {len(audio_files)} audio files")
        
        # Perform batch prediction
        results = batch_predict(audio_files, artifacts)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Display results
        print("\nResults:")
        print(df.to_string(index=False))
        
        # Save to CSV if requested
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nResults saved to {args.out}")
    
    else:
        print("Please provide either --file or --dir argument")
