# Spoken Digit Classifier

## Overview

This is a lightweight audio classification system that recognizes spoken digits (0-9) using machine learning. The system processes audio files through MFCC (Mel-Frequency Cepstral Coefficients) feature extraction and classifies them using classical machine learning models like SVM, Logistic Regression, or XGBoost. Built with a focus on simplicity, speed, and accuracy, the application uses the Free Spoken Digit Dataset and provides both batch inference and interactive evaluation capabilities through a Streamlit web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components

**Feature Extraction Pipeline**: The system uses a two-stage audio processing approach. First, the AudioProcessor handles audio loading, resampling to 8kHz, normalization, and silence trimming. Then, the MFCCExtractor computes MFCC coefficients along with delta and delta-delta features, followed by StandardScaler normalization for consistent feature scaling.

**Classical ML Models**: Instead of deep learning, the architecture employs classical machine learning through the ClassicalModel wrapper. This supports multiple algorithms (SVM with RBF/Linear kernels, Logistic Regression, XGBoost) with grid search hyperparameter tuning. This choice prioritizes fast inference (<200ms) and small model size (<5MB).

**Speaker-Holdout Validation**: The FSIDDataset implements speaker-holdout data splitting to ensure models generalize to unseen speakers, preventing speaker-specific overfitting that could inflate performance metrics.

**Modular Design**: The codebase separates concerns across distinct modules - data handling, feature extraction, model training/evaluation, and utilities. Each component can be independently tested and modified.

### Data Flow

Audio files → AudioProcessor (load/preprocess) → MFCCExtractor (feature extraction) → StandardScaler (normalization) → ClassicalModel (prediction) → Results

### Performance Optimization

**Timing Infrastructure**: Custom Timer context manager and benchmarking utilities measure execution times at both component and end-to-end levels. The system tracks average and 95th percentile inference times.

**Artifact Management**: Comprehensive save/load system for models, scalers, configurations, and evaluation metrics enables reproducible experiments and deployment.

**Synthetic Data Fallback**: When actual audio files are unavailable, the system generates synthetic audio signals for demonstration purposes.

## External Dependencies

**Core ML Stack**: scikit-learn for classical models and preprocessing, librosa for audio processing and MFCC extraction, numpy/pandas for data manipulation.

**Audio Processing**: soundfile for audio I/O operations, librosa for digital signal processing and feature extraction.

**Optional Enhancements**: XGBoost for gradient boosting models (gracefully handles absence), datasets library for Hugging Face integration.

**Web Interface**: Streamlit for interactive web application, matplotlib/seaborn for visualization and evaluation plots.

**Dataset Source**: Free Spoken Digit Dataset via Hugging Face datasets library, with local caching and metadata management.

**Model Persistence**: joblib for efficient model serialization, JSON for configuration storage.