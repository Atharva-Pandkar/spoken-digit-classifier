import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import joblib
from scipy.fftpack import dct
import scipy.signal

class MFCCExtractor:
    """Extract MFCC features with delta and delta-delta coefficients"""
    
    def __init__(self, 
                 n_mfcc: int = 20,
                 n_fft: int = 512,
                 hop_length: int = 80,  # 10ms at 8kHz
                 n_mels: int = 40,
                 use_deltas: bool = True,
                 use_delta_deltas: bool = True):
        
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_deltas = use_deltas
        self.use_delta_deltas = use_delta_deltas
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features from audio"""
        # Extract MFCC coefficients using manual implementation
        mfccs = self._compute_mfcc(audio, sr)
        
        features = [mfccs]
        
        # Add delta features
        if self.use_deltas:
            delta_mfccs = self._compute_deltas(mfccs)
            features.append(delta_mfccs)
        
        # Add delta-delta features
        if self.use_delta_deltas:
            delta2_mfccs = self._compute_deltas(self._compute_deltas(mfccs))
            features.append(delta2_mfccs)
        
        # Concatenate all features
        features = np.vstack(features)
        
        return features
    
    def aggregate_features(self, features: np.ndarray) -> np.ndarray:
        """Aggregate time-varying features to fixed-length vector"""
        # Use statistical aggregation: mean and standard deviation
        mean_features = np.mean(features, axis=1)
        std_features = np.std(features, axis=1)
        
        # Concatenate mean and std
        aggregated = np.concatenate([mean_features, std_features])
        
        return aggregated
    
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract and aggregate MFCC features"""
        # Extract MFCC features
        mfcc_features = self.extract_mfcc(audio, sr)
        
        # Aggregate to fixed-length vector
        feature_vector = self.aggregate_features(mfcc_features)
        
        return feature_vector
    
    def fit_scaler(self, features: np.ndarray):
        """Fit the feature scaler"""
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return self.scaler.transform(features)
    
    def fit_transform_features(self, features: np.ndarray) -> np.ndarray:
        """Fit scaler and transform features"""
        self.fit_scaler(features)
        return self.transform_features(features)
    
    def get_feature_dimension(self) -> int:
        """Get the expected feature dimension"""
        base_dim = self.n_mfcc
        
        if self.use_deltas:
            base_dim += self.n_mfcc
        
        if self.use_delta_deltas:
            base_dim += self.n_mfcc
        
        # Multiply by 2 for mean and std aggregation
        return base_dim * 2
    
    def _compute_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute MFCC features manually"""
        # Compute spectrogram
        f, t, Sxx = scipy.signal.spectrogram(
            audio, fs=sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length
        )
        
        # Convert to mel scale (simplified)
        mel_filters = self._mel_filter_bank(f, sr)
        mel_spec = np.dot(mel_filters, Sxx)
        
        # Avoid log(0)
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel_spec = np.log(mel_spec)
        
        # Apply DCT to get MFCC
        mfccs = dct(log_mel_spec, type=2, axis=0, norm='ortho')[:self.n_mfcc, :]
        
        return mfccs
    
    def _mel_filter_bank(self, freqs: np.ndarray, sr: int) -> np.ndarray:
        """Create mel filter bank"""
        # Simplified mel filter bank
        n_fft_half = len(freqs)
        mel_filters = np.zeros((self.n_mels, n_fft_half))
        
        # Create triangular filters (simplified)
        mel_points = np.linspace(0, n_fft_half - 1, self.n_mels + 2)
        
        for i in range(self.n_mels):
            left = int(mel_points[i])
            center = int(mel_points[i + 1])
            right = int(mel_points[i + 2])
            
            # Left slope
            for j in range(left, center):
                if center > left:
                    mel_filters[i, j] = (j - left) / (center - left)
            
            # Right slope
            for j in range(center, right):
                if right > center:
                    mel_filters[i, j] = (right - j) / (right - center)
        
        return mel_filters
    
    def _compute_deltas(self, features: np.ndarray, N: int = 2) -> np.ndarray:
        """Compute delta features"""
        if features.shape[1] < 3:
            return np.zeros_like(features)
        
        deltas = np.zeros_like(features)
        padded = np.pad(features, ((0, 0), (N, N)), mode='edge')
        
        for t in range(features.shape[1]):
            deltas[:, t] = np.sum([
                n * (padded[:, t + N + n] - padded[:, t + N - n])
                for n in range(1, N + 1)
            ], axis=0) / (2 * sum(n**2 for n in range(1, N + 1)))
        
        return deltas
    
    def save_scaler(self, filepath: str):
        """Save the fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")
        
        joblib.dump(self.scaler, filepath)
    
    def load_scaler(self, filepath: str):
        """Load a fitted scaler"""
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
    
    def get_config(self) -> Dict:
        """Get extractor configuration"""
        return {
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'use_deltas': self.use_deltas,
            'use_delta_deltas': self.use_delta_deltas,
            'feature_dim': self.get_feature_dimension()
        }
