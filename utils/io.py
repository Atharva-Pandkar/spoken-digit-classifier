import json
import joblib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
import os

def save_artifacts(artifacts: Dict[str, Any], output_dir: str):
    """Save all model artifacts to directory"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if 'model' in artifacts:
        model_path = output_path / 'model.pkl'
        joblib.dump(artifacts['model'], model_path)
    
    # Save feature scaler
    if 'scaler' in artifacts:
        scaler_path = output_path / 'scaler.pkl'
        joblib.dump(artifacts['scaler'], scaler_path)
    
    # Save feature extractor configuration
    if 'feature_config' in artifacts:
        config_path = output_path / 'feature_config.json'
        with open(config_path, 'w') as f:
            json.dump(artifacts['feature_config'], f, indent=2)
    
    # Save evaluation metrics
    metrics = {}
    for key in ['train_accuracy', 'val_accuracy', 'test_accuracy', 
                'avg_inference_time_ms', 'p95_inference_time_ms']:
        if key in artifacts:
            metrics[key] = float(artifacts[key])
    
    # Save confusion matrix
    if 'confusion_matrix' in artifacts:
        metrics['confusion_matrix'] = artifacts['confusion_matrix'].tolist()
    
    # Save classification report
    if 'classification_report' in artifacts:
        metrics['classification_report'] = artifacts['classification_report']
    
    # Save per-digit accuracy
    if 'per_digit_accuracy' in artifacts:
        metrics['per_digit_accuracy'] = artifacts['per_digit_accuracy']
    
    metrics_path = output_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training configuration
    if 'training_config' in artifacts:
        config_path = output_path / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(artifacts['training_config'], f, indent=2)
    
    print(f"Artifacts saved to {output_path}")

def load_artifacts(artifact_dir: str) -> Dict[str, Any]:
    """Load all model artifacts from directory"""
    artifact_path = Path(artifact_dir)
    
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
    
    artifacts = {}
    
    # Load model
    model_path = artifact_path / 'model.pkl'
    if model_path.exists():
        artifacts['model'] = joblib.load(model_path)
    
    # Load scaler
    scaler_path = artifact_path / 'scaler.pkl'
    if scaler_path.exists():
        artifacts['scaler'] = joblib.load(scaler_path)
    
    # Load feature config
    config_path = artifact_path / 'feature_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            artifacts['feature_config'] = json.load(f)
    
    # Load metrics
    metrics_path = artifact_path / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            artifacts.update(metrics)
    
    # Load training config
    training_config_path = artifact_path / 'training_config.json'
    if training_config_path.exists():
        with open(training_config_path, 'r') as f:
            artifacts['training_config'] = json.load(f)
    
    return artifacts

def save_json(data: Dict[str, Any], filepath: str):
    """Save dictionary as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file as dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes"""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    return 0.0
