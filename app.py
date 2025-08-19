import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import time
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optional audio imports
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    AUDIO_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    AUDIO_AVAILABLE = False
    print(f"Audio functionality disabled: {e}")

# Import our modules
from data.dataset import FSIDDataset
from features.audio_io import AudioProcessor
from features.mfcc import MFCCExtractor
from features.mel_spectrogram import MelSpectrogramExtractor
from models.classical import ClassicalModel
from models.cnn_model import CNNModel
from train import train_model
from eval import evaluate_model
from infer import predict_digit
from utils.timing import time_function
from utils.io import save_artifacts, load_artifacts

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_artifacts' not in st.session_state:
    st.session_state.model_artifacts = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'recording' not in st.session_state:
    st.session_state.recording = False

def main():
    st.title("🎤 Spoken Digit Classifier")
    st.markdown("**Audio → Digit Classification using MFCC features and Classical ML**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dataset & Training", "Model Evaluation", "Inference", "Live Microphone (Optional)"]
    )
    
    if page == "Dataset & Training":
        dataset_training_page()
    elif page == "Model Evaluation":
        evaluation_page()
    elif page == "Inference":
        inference_page()
    elif page == "Live Microphone (Optional)":
        microphone_page()

def dataset_training_page():
    st.header("📊 Dataset & Model Training")
    
    # Dataset loading section
    st.subheader("1. Dataset Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load FSDD Dataset"):
            with st.spinner("Loading Free Spoken Digit Dataset..."):
                try:
                    dataset = FSIDDataset()
                    dataset.load_dataset()
                    st.session_state.dataset = dataset
                    st.success(f"✅ Dataset loaded successfully!")
                    st.info(f"Total samples: {len(dataset.metadata)}")
                    
                    # Display dataset statistics
                    speaker_counts = dataset.metadata['speaker'].value_counts()
                    digit_counts = dataset.metadata['digit'].value_counts()
                    
                    st.write("**Speaker distribution:**")
                    st.bar_chart(speaker_counts)
                    
                    st.write("**Digit distribution:**")
                    st.bar_chart(digit_counts)
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
    
    with col2:
        if st.session_state.dataset is not None:
            st.metric("Total Samples", len(st.session_state.dataset.metadata))
            st.metric("Unique Speakers", st.session_state.dataset.metadata['speaker'].nunique())
            st.metric("Digits (0-9)", st.session_state.dataset.metadata['digit'].nunique())
    
    # Training section
    st.subheader("2. Model Training")
    
    if st.session_state.dataset is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Choose model type:",
                ["SVM (RBF)", "SVM (Linear)", "Logistic Regression", "XGBoost", "CNN"]
            )
        
        with col2:
            test_size = st.slider("Test size", 0.1, 0.3, 0.2, 0.05)
        
        with col3:
            val_size = st.slider("Validation size", 0.1, 0.3, 0.15, 0.05)
        
        # Advanced options
        with st.expander("🔧 Advanced Training Options"):
            if model_type == "CNN":
                st.write("**CNN Model Parameters:**")
                epochs = st.slider("Training Epochs", 10, 100, 50)
                batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
                st.info("CNN uses mel-spectrograms (60×80) as input features")
            else:
                st.write("**Classical Model Parameters:**")
                mfcc_coeffs = st.slider("MFCC Coefficients", 10, 30, 20)
                use_deltas = st.checkbox("Use Delta coefficients", True)
                use_delta_deltas = st.checkbox("Use Delta-Delta coefficients", True)
            
            random_seed = st.number_input("Random Seed", 0, 999, 42)
        
        if st.button("🚀 Train Model"):
            if model_type == "CNN":
                st.write("### Training CNN Model")
                st.info("📊 Training progress will be shown below:")
                
            with st.spinner("Training model... This may take a few minutes." if model_type != "CNN" else "Initializing CNN training..."):
                try:
                    # Prepare training configuration
                    config = {
                        'model_type': model_type,
                        'test_size': test_size,
                        'val_size': val_size,
                        'random_seed': random_seed
                    }
                    
                    if model_type == "CNN":
                        config.update({
                            'epochs': epochs,
                            'batch_size': batch_size
                        })
                    else:
                        config.update({
                            'mfcc_coeffs': mfcc_coeffs,
                            'use_deltas': use_deltas,
                            'use_delta_deltas': use_delta_deltas
                        })
                    
                    # Train the model
                    artifacts = train_model(st.session_state.dataset, config)
                    st.session_state.model_artifacts = artifacts
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display training results
                    st.subheader("Training Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Accuracy", f"{artifacts['train_accuracy']:.3f}")
                    with col2:
                        st.metric("Validation Accuracy", f"{artifacts['val_accuracy']:.3f}")
                    with col3:
                        st.metric("Model Size", f"{artifacts['model_size_mb']:.2f} MB")
                    
                    # Display training time and additional info for CNN
                    if model_type == "CNN":
                        epochs_trained = artifacts.get('epochs_trained', epochs)
                        st.info(f"⏱️ Training completed in {artifacts['training_time']:.2f} seconds ({epochs_trained} epochs)")
                        if 'history' in artifacts and 'val_accuracy' in artifacts['history']:
                            # Plot training history for CNN
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Accuracy plot
                            history = artifacts['history']
                            ax1.plot(history['accuracy'], label='Training Accuracy')
                            if 'val_accuracy' in history:
                                ax1.plot(history['val_accuracy'], label='Validation Accuracy')
                            ax1.set_title('Model Accuracy')
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Accuracy')
                            ax1.legend()
                            
                            # Loss plot
                            ax2.plot(history['loss'], label='Training Loss')
                            if 'val_loss' in history:
                                ax2.plot(history['val_loss'], label='Validation Loss')
                            ax2.set_title('Model Loss')
                            ax2.set_xlabel('Epoch')
                            ax2.set_ylabel('Loss')
                            ax2.legend()
                            
                            st.pyplot(fig)
                    else:
                        st.info(f"⏱️ Training completed in {artifacts['training_time']:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.exception(e)
    else:
        st.warning("⚠️ Please load the dataset first before training.")

def evaluation_page():
    st.header("📈 Model Evaluation")
    
    if not st.session_state.model_trained or st.session_state.model_artifacts is None:
        st.warning("⚠️ Please train a model first in the 'Dataset & Training' page.")
        return
    
    artifacts = st.session_state.model_artifacts
    
    # Test evaluation
    st.subheader("Test Set Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", f"{artifacts['test_accuracy']:.3f}")
    with col2:
        st.metric("Avg Inference Time", f"{artifacts['avg_inference_time_ms']:.1f} ms")
    with col3:
        st.metric("95th Percentile Latency", f"{artifacts['p95_inference_time_ms']:.1f} ms")
    with col4:
        success_icon = "✅" if artifacts['test_accuracy'] >= 0.95 else "❌"
        latency_icon = "✅" if artifacts['avg_inference_time_ms'] <= 200 else "❌"
        st.write(f"Accuracy Target: {success_icon}")
        st.write(f"Latency Target: {latency_icon}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        artifacts['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax
    )
    ax.set_xlabel('Predicted Digit')
    ax.set_ylabel('True Digit')
    ax.set_title('Confusion Matrix - Test Set')
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("Detailed Classification Report")
    
    if 'classification_report' in artifacts:
        report_df = pd.DataFrame(artifacts['classification_report']).transpose()
        st.dataframe(report_df.round(3))
    
    # Per-digit performance
    st.subheader("Per-Digit Performance")
    
    if 'per_digit_accuracy' in artifacts:
        digit_perf = artifacts['per_digit_accuracy']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        digits = list(digit_perf.keys())
        accuracies = list(digit_perf.values())
        
        bars = ax.bar(digits, accuracies, color='lightblue', edgecolor='navy', alpha=0.7)
        ax.set_xlabel('Digit')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Digit Classification Accuracy')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{acc:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    # Latency Analysis
    st.subheader("Latency Analysis")
    
    if 'latency_distribution' in artifacts:
        latencies = artifacts['latency_distribution']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(latencies, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inference Time Distribution')
        ax1.axvline(200, color='red', linestyle='--', label='200ms Target')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(latencies)
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Box Plot')
        ax2.axhline(200, color='red', linestyle='--', label='200ms Target')
        ax2.legend()
        
        st.pyplot(fig)
    
    # Download artifacts
    st.subheader("📥 Download Model Artifacts")
    
    if st.button("Generate Download Package"):
        with st.spinner("Preparing download package..."):
            try:
                # Create a temporary directory for artifacts
                with tempfile.TemporaryDirectory() as temp_dir:
                    artifact_path = Path(temp_dir) / "model_artifacts"
                    artifact_path.mkdir()
                    
                    # Save artifacts
                    save_artifacts(artifacts, artifact_path)
                    
                    # Create a zip file
                    import zipfile
                    zip_path = Path(temp_dir) / "spoken_digit_classifier_artifacts.zip"
                    
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file_path in artifact_path.rglob('*'):
                            if file_path.is_file():
                                zipf.write(file_path, file_path.relative_to(artifact_path))
                    
                    # Read the zip file for download
                    with open(zip_path, 'rb') as f:
                        zip_data = f.read()
                    
                    st.download_button(
                        label="📦 Download Model Package",
                        data=zip_data,
                        file_name="spoken_digit_classifier_artifacts.zip",
                        mime="application/zip"
                    )
                    
                    st.success("✅ Download package ready!")
                    
            except Exception as e:
                st.error(f"❌ Error creating download package: {str(e)}")

def inference_page():
    st.header("🎯 Model Inference")
    
    if not st.session_state.model_trained or st.session_state.model_artifacts is None:
        st.warning("⚠️ Please train a model first in the 'Dataset & Training' page.")
        return
    
    # File upload section
    st.subheader("📁 Upload Audio Files")
    
    uploaded_files = st.file_uploader(
        "Choose WAV files",
        accept_multiple_files=True,
        type=['wav']
    )
    
    if uploaded_files:
        st.subheader("🔍 Inference Results")
        
        results = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Perform inference
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    start_time = time.time()
                    prediction, confidence = predict_digit(tmp_path, st.session_state.model_artifacts)
                    inference_time = (time.time() - start_time) * 1000
                
                results.append({
                    'Filename': uploaded_file.name,
                    'Predicted Digit': prediction,
                    'Confidence': f"{confidence:.3f}",
                    'Inference Time (ms)': f"{inference_time:.1f}"
                })
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                results.append({
                    'Filename': uploaded_file.name,
                    'Predicted Digit': 'Error',
                    'Confidence': '0.000',
                    'Inference Time (ms)': 'N/A'
                })
        
        # Display results
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            valid_results = [r for r in results if r['Predicted Digit'] != 'Error']
            
            if valid_results:
                st.subheader("📊 Batch Inference Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Files Processed", len(results))
                
                with col2:
                    successful = len(valid_results)
                    st.metric("Successful Predictions", successful)
                
                with col3:
                    if valid_results:
                        avg_time = np.mean([float(r['Inference Time (ms)']) for r in valid_results if r['Inference Time (ms)'] != 'N/A'])
                        st.metric("Avg Inference Time", f"{avg_time:.1f} ms")
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="inference_results.csv",
                    mime="text/csv"
                )
    
    # Sample audio section
    st.subheader("🎵 Try with Sample Audio")
    model_info = "MFCC features" if not st.session_state.model_artifacts.get('is_cnn', False) else "mel-spectrogram"
    st.info(f"💡 Upload your own WAV files (8 kHz recommended) or record audio saying digits 0-9. Current model uses {model_info} for classification.")

def microphone_page():
    st.header("🎤 Live Microphone Recording")
    
    # Check if audio libraries are available
    if not AUDIO_AVAILABLE:
        st.error("🔇 **Microphone functionality not available**")
        st.info("""
        **Audio libraries couldn't be loaded in this environment.**
        
        **Alternative Options:**
        1. Use the **"Inference"** tab to upload pre-recorded audio files
        2. Record audio on your device and save as .wav file, then upload
        
        **For developers**: Install PortAudio system library to enable microphone support.
        """)
        return
    
    if not st.session_state.model_trained or st.session_state.model_artifacts is None:
        st.warning("⚠️ Please train a model first in the 'Dataset & Training' page.")
        return
    
    # Show current model info
    model_type = st.session_state.model_artifacts.get('config', {}).get('model_type', 'Unknown')
    test_accuracy = st.session_state.model_artifacts.get('test_accuracy', 0.0)
    st.info(f"🤖 Current Model: **{model_type}** (Test Accuracy: {test_accuracy:.1%})")
    
    st.markdown("""
    ### 🎙️ Record Your Voice
    **Instructions:**
    1. Click "Start Recording" 
    2. **Speak a single digit (0-9) clearly**
    3. Click "Stop Recording" 
    4. Get instant prediction!
    """)
    
    # Recording parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        duration = st.slider("Recording Duration (seconds)", 1, 5, 2)
    with col2:
        sample_rate = 8000  # Fixed for model compatibility
        st.metric("Sample Rate", f"{sample_rate} Hz")
    with col3:
        st.metric("Status", "Ready" if not st.session_state.get('recording', False) else "Recording...")
    
    # Recording controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🎙️ Start Recording", type="primary", disabled=st.session_state.get('recording', False)):
            try:
                st.session_state.recording = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start recording: {e}")
    
    with col2:
        if st.button("⏹️ Stop & Predict", disabled=not st.session_state.get('recording', False)):
            st.session_state.recording = False
            st.rerun()
    
    # Recording status and process
    if st.session_state.get('recording', False):
        st.error("🔴 **RECORDING IN PROGRESS**")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start recording
        try:
            st.info(f"🎤 Recording for {duration} seconds... **Speak a digit (0-9) now!**")
            
            # Record audio
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
            
            # Show progress
            for i in range(duration * 10):  # 10 updates per second
                time.sleep(0.1)
                progress = (i + 1) / (duration * 10)
                progress_bar.progress(progress)
                status_text.text(f"Recording... {((i + 1) / 10):.1f}s / {duration}s")
            
            sd.wait()  # Wait until recording is finished
            
            # Save temporary audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name
                write(temp_path, sample_rate, (audio_data * 32767).astype(np.int16))
            
            # Process and predict
            st.session_state.recording = False
            progress_bar.progress(1.0)
            status_text.text("Processing and predicting...")
            
            # Make prediction
            with st.spinner("🧠 Analyzing your voice..."):
                prediction_result = predict_digit(temp_path, st.session_state.model_artifacts)
                
            # Clean up temp file
            os.unlink(temp_path)
            
            # Display results
            st.success("✅ Recording completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="🎯 Predicted Digit",
                    value=str(prediction_result['predicted_digit']),
                    delta=f"Confidence: {prediction_result.get('confidence', 0):.1%}"
                )
            
            with col2:
                processing_time = prediction_result.get('processing_time', 0)
                st.metric(
                    label="⚡ Processing Time",
                    value=f"{processing_time:.1f} ms"
                )
            
            # Show confidence for all digits
            if 'probabilities' in prediction_result:
                st.subheader("📊 Confidence Scores")
                probs = prediction_result['probabilities']
                prob_data = pd.DataFrame({
                    'Digit': range(10),
                    'Confidence': probs if hasattr(probs, '__len__') and len(probs) == 10 else [0] * 10
                })
                st.bar_chart(prob_data.set_index('Digit'))
            
            st.rerun()
            
        except Exception as e:
            st.session_state.recording = False
            st.error(f"❌ Recording failed: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**\n- Check microphone permissions\n- Ensure microphone is connected\n- Try refreshing the page")
    
    else:
        # Show example audio visualization
        st.subheader("🎵 Audio Tips")
        st.info("""
        **For best results:**
        - Speak clearly and at normal volume
        - Say only one digit (0, 1, 2, 3, 4, 5, 6, 7, 8, or 9)
        - Avoid background noise
        - Wait for the full recording duration
        """)
        
        # Test microphone
        if st.button("🔊 Test Microphone (2 sec)"):
            try:
                with st.spinner("Testing microphone..."):
                    test_audio = sd.rec(int(2 * sample_rate), samplerate=sample_rate, channels=1)
                    sd.wait()
                    
                    # Check if audio was recorded
                    max_amplitude = np.max(np.abs(test_audio))
                    if max_amplitude > 0.001:
                        st.success(f"✅ Microphone working! Max amplitude: {max_amplitude:.3f}")
                    else:
                        st.warning("⚠️ Very low audio signal. Check microphone volume.")
                        
            except Exception as e:
                st.error(f"❌ Microphone test failed: {e}")

if __name__ == "__main__":
    main()
