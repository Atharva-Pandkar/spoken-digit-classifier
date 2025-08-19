import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data.dataset import FSIDDataset
from models.classical import ClassicalModel
from features.mfcc import MFCCExtractor
from features.audio_io import AudioProcessor
from utils.timing import benchmark_function

def evaluate_model(artifacts: Dict[str, Any], 
                  dataset: Optional[FSIDDataset] = None,
                  test_with_noise: bool = True) -> Dict[str, Any]:
    """Comprehensive model evaluation"""
    
    print("Starting comprehensive model evaluation...")
    
    # Extract components from artifacts
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_config = artifacts['feature_config']
    
    # Initialize components
    audio_processor = AudioProcessor(target_sr=8000)
    feature_extractor = MFCCExtractor(
        n_mfcc=feature_config['n_mfcc'],
        use_deltas=feature_config['use_deltas'],
        use_delta_deltas=feature_config['use_delta_deltas']
    )
    feature_extractor.scaler = scaler
    feature_extractor.is_fitted = True
    
    evaluation_results = {}
    
    # Basic performance metrics (already in artifacts)
    evaluation_results.update({
        'test_accuracy': artifacts.get('test_accuracy', 0.0),
        'confusion_matrix': artifacts.get('confusion_matrix', np.zeros((10, 10))),
        'classification_report': artifacts.get('classification_report', {}),
        'per_digit_accuracy': artifacts.get('per_digit_accuracy', {}),
        'avg_inference_time_ms': artifacts.get('avg_inference_time_ms', 0.0),
        'p95_inference_time_ms': artifacts.get('p95_inference_time_ms', 0.0)
    })
    
    # Noise robustness testing
    if test_with_noise and dataset is not None:
        print("Testing noise robustness...")
        noise_results = test_noise_robustness(
            model, feature_extractor, audio_processor, dataset
        )
        evaluation_results['noise_robustness'] = noise_results
    
    # Model complexity analysis
    print("Analyzing model complexity...")
    complexity_analysis = analyze_model_complexity(artifacts)
    evaluation_results['model_complexity'] = complexity_analysis
    
    # Performance analysis
    print("Performing detailed performance analysis...")
    performance_analysis = analyze_performance(artifacts)
    evaluation_results['performance_analysis'] = performance_analysis
    
    return evaluation_results

def test_noise_robustness(model: ClassicalModel, 
                         feature_extractor: MFCCExtractor,
                         audio_processor: AudioProcessor,
                         dataset: FSIDDataset,
                         noise_levels: list = [20, 15, 10]) -> Dict[str, Any]:
    """Test model robustness to noise"""
    
    # Get test split
    if dataset.splits is None:
        dataset.create_speaker_holdout_split()
    
    test_data = dataset.splits['test']
    
    noise_results = {}
    
    for snr_db in noise_levels:
        print(f"Testing with SNR {snr_db} dB...")
        
        accuracies = []
        
        for _, row in test_data.iterrows():
            try:
                # Load and preprocess audio
                audio = audio_processor.preprocess_audio(row['filepath'])
                
                # Add noise
                noisy_audio = add_noise(audio, snr_db)
                
                # Extract features
                features = feature_extractor.extract_features(noisy_audio, 8000)
                features_scaled = feature_extractor.transform_features(features.reshape(1, -1))
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                true_label = row['digit']
                
                accuracies.append(prediction == true_label)
                
            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")
                continue
        
        if accuracies:
            noise_results[f'snr_{snr_db}db'] = {
                'accuracy': np.mean(accuracies),
                'num_samples': len(accuracies)
            }
    
    return noise_results

def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to audio signal"""
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    
    # Add noise to signal
    noisy_audio = audio + noise
    
    return noisy_audio

def analyze_model_complexity(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze model complexity and size"""
    
    complexity = {
        'model_type': artifacts.get('model_type', 'Unknown'),
        'model_size_mb': artifacts.get('model_size_mb', 0.0),
        'feature_dimension': artifacts.get('feature_dimension', 0),
        'training_time_seconds': artifacts.get('training_time', 0.0)
    }
    
    # Calculate complexity score (simple heuristic)
    size_score = min(artifacts.get('model_size_mb', 0.0) / 5.0, 1.0)  # Normalize by 5MB target
    time_score = min(artifacts.get('training_time', 0.0) / 300.0, 1.0)  # Normalize by 5 minutes
    
    complexity['complexity_score'] = (size_score + time_score) / 2.0
    
    return complexity

def analyze_performance(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze detailed performance characteristics"""
    
    # Extract performance metrics
    test_accuracy = artifacts.get('test_accuracy', 0.0)
    avg_inference_time = artifacts.get('avg_inference_time_ms', 0.0)
    p95_inference_time = artifacts.get('p95_inference_time_ms', 0.0)
    
    # Performance scores
    accuracy_score = test_accuracy  # Already normalized 0-1
    latency_score = max(0, min(1, (200 - avg_inference_time) / 200))  # 200ms target
    
    # Overall performance score
    overall_score = (accuracy_score * 0.7 + latency_score * 0.3)  # Weight accuracy more
    
    # Performance categorization
    if overall_score >= 0.9:
        performance_category = "Excellent"
    elif overall_score >= 0.8:
        performance_category = "Good"
    elif overall_score >= 0.7:
        performance_category = "Fair"
    else:
        performance_category = "Poor"
    
    # Target achievement
    accuracy_target_met = test_accuracy >= 0.95
    latency_target_met = avg_inference_time <= 200
    
    analysis = {
        'accuracy_score': accuracy_score,
        'latency_score': latency_score,
        'overall_score': overall_score,
        'performance_category': performance_category,
        'accuracy_target_met': accuracy_target_met,
        'latency_target_met': latency_target_met,
        'targets_met': accuracy_target_met and latency_target_met
    }
    
    return analysis

def generate_evaluation_report(evaluation_results: Dict[str, Any]) -> str:
    """Generate a comprehensive evaluation report"""
    
    report = []
    report.append("# Spoken Digit Classifier - Evaluation Report\n")
    
    # Basic Performance
    report.append("## Basic Performance Metrics")
    report.append(f"- Test Accuracy: {evaluation_results['test_accuracy']:.3f}")
    report.append(f"- Average Inference Time: {evaluation_results['avg_inference_time_ms']:.1f} ms")
    report.append(f"- 95th Percentile Inference Time: {evaluation_results['p95_inference_time_ms']:.1f} ms")
    report.append("")
    
    # Target Achievement
    if 'performance_analysis' in evaluation_results:
        perf = evaluation_results['performance_analysis']
        report.append("## Target Achievement")
        report.append(f"- Accuracy ≥95%: {'✅' if perf['accuracy_target_met'] else '❌'}")
        report.append(f"- Latency ≤200ms: {'✅' if perf['latency_target_met'] else '❌'}")
        report.append(f"- Overall Performance: {perf['performance_category']}")
        report.append("")
    
    # Model Complexity
    if 'model_complexity' in evaluation_results:
        comp = evaluation_results['model_complexity']
        report.append("## Model Complexity")
        report.append(f"- Model Type: {comp['model_type']}")
        report.append(f"- Model Size: {comp['model_size_mb']:.2f} MB")
        report.append(f"- Feature Dimension: {comp['feature_dimension']}")
        report.append(f"- Training Time: {comp['training_time_seconds']:.2f} seconds")
        report.append("")
    
    # Noise Robustness
    if 'noise_robustness' in evaluation_results:
        noise = evaluation_results['noise_robustness']
        report.append("## Noise Robustness")
        for snr_level, results in noise.items():
            snr_db = snr_level.replace('snr_', '').replace('db', '')
            report.append(f"- SNR {snr_db} dB: {results['accuracy']:.3f} accuracy")
        report.append("")
    
    # Per-digit Performance
    if 'per_digit_accuracy' in evaluation_results:
        report.append("## Per-Digit Performance")
        per_digit = evaluation_results['per_digit_accuracy']
        for digit, accuracy in sorted(per_digit.items()):
            report.append(f"- Digit {digit}: {accuracy:.3f}")
        report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Example usage
    from utils.io import load_artifacts
    
    # Load artifacts
    artifacts = load_artifacts("./model_artifacts")
    
    # Load dataset for noise testing
    dataset = FSIDDataset()
    dataset.load_dataset()
    
    # Evaluate model
    evaluation_results = evaluate_model(artifacts, dataset)
    
    # Generate report
    report = generate_evaluation_report(evaluation_results)
    print(report)
    
    # Save report
    with open("evaluation_report.md", "w") as f:
        f.write(report)
