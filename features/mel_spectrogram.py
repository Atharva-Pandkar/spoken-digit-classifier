import numpy as np
from scipy import signal
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import joblib

class MelSpectrogramExtractor:
    """Extract mel-spectrogram features for CNN input"""
    
    def __init__(self,
                 n_mels: int = 60,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 target_length: int = 80,
                 fmin: float = 0.0,
                 fmax: float = 4000.0):
        
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        self.fmin = fmin
        self.fmax = fmax
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def hz_to_mel(self, hz):
        """Convert Hz to mel scale"""
        return 2595 * np.log10(1 + hz / 700.0)
    
    def mel_to_hz(self, mel):
        """Convert mel scale to Hz"""
        return 700 * (10**(mel / 2595) - 1)
    
    def create_mel_filter_bank(self, sr: int, n_fft_bins: int = None):
        """Create mel filter bank"""
        # Use actual number of FFT bins from the spectrogram
        if n_fft_bins is None:
            n_bins = self.n_fft // 2 + 1
        else:
            n_bins = n_fft_bins
        
        # Create frequency bins
        freqs = np.linspace(0, sr // 2, n_bins)
        
        # Convert to mel scale
        mel_min = self.hz_to_mel(self.fmin)
        mel_max = self.hz_to_mel(min(self.fmax, sr // 2))
        
        # Create mel points
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        
        # Convert Hz points to FFT bin indices
        bin_points = np.floor((n_bins - 1) * hz_points / (sr // 2)).astype(int)
        bin_points = np.clip(bin_points, 0, n_bins - 1)
        
        # Create filter bank
        mel_filters = np.zeros((self.n_mels, n_bins))
        
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Ensure indices are valid
            left = max(0, min(left, n_bins - 1))
            center = max(0, min(center, n_bins - 1))
            right = max(0, min(right, n_bins - 1))
            
            # Left slope
            if center > left:
                for j in range(left, center):
                    mel_filters[i, j] = (j - left) / (center - left)
            
            # Right slope
            if right > center:
                for j in range(center, right):
                    mel_filters[i, j] = (right - j) / (right - center)
        
        return mel_filters
    
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract mel-spectrogram from audio"""
        
        # Ensure minimum audio length
        min_length = self.n_fft
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        # Compute STFT with overlap check
        noverlap = min(self.n_fft - self.hop_length, self.n_fft - 1)
        
        try:
            f, t, Zxx = signal.stft(
                audio, 
                fs=sr, 
                nperseg=self.n_fft,
                noverlap=noverlap,
                window='hann'
            )
        except ValueError as e:
            # If STFT fails, use smaller window
            nperseg = min(self.n_fft, len(audio))
            noverlap = min(noverlap, nperseg - 1)
            f, t, Zxx = signal.stft(
                audio, 
                fs=sr, 
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
        
        # Get magnitude spectrogram
        magnitude = np.abs(Zxx)
        
        # Apply mel filter bank
        mel_filters = self.create_mel_filter_bank(sr, magnitude.shape[0])
        mel_spec = np.dot(mel_filters, magnitude)
        
        # Convert to log scale
        log_mel_spec = np.log(mel_spec + 1e-10)  # Add small epsilon to avoid log(0)
        
        return log_mel_spec
    
    def pad_or_truncate(self, mel_spec: np.ndarray) -> np.ndarray:
        """Pad or truncate mel-spectrogram to target length"""
        current_length = mel_spec.shape[1]
        
        if current_length < self.target_length:
            # Pad with zeros
            pad_width = self.target_length - current_length
            padded = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            return padded
        elif current_length > self.target_length:
            # Truncate
            return mel_spec[:, :self.target_length]
        else:
            return mel_spec
    
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract mel-spectrogram features for CNN input"""
        
        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio, sr)
        
        # Pad or truncate to target length
        mel_spec = self.pad_or_truncate(mel_spec)
        
        # Add channel dimension and return as (height, width, channels)
        mel_spec = mel_spec[..., np.newaxis]  # Add channel dimension
        
        return mel_spec
    
    def fit_scaler(self, features: np.ndarray):
        """Fit the feature scaler"""
        # Flatten features for scaling
        n_samples, height, width, channels = features.shape
        flattened = features.reshape(n_samples, -1)
        
        self.scaler.fit(flattened)
        self.is_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        original_shape = features.shape
        
        if features.ndim == 3:
            # Single sample
            flattened = features.reshape(1, -1)
            scaled_flat = self.scaler.transform(flattened)
            return scaled_flat.reshape(original_shape)
        else:
            # Multiple samples
            n_samples = features.shape[0]
            flattened = features.reshape(n_samples, -1)
            scaled_flat = self.scaler.transform(flattened)
            return scaled_flat.reshape(original_shape)
    
    def fit_transform_features(self, features: np.ndarray) -> np.ndarray:
        """Fit scaler and transform features"""
        self.fit_scaler(features)
        return self.transform_features(features)
    
    def get_feature_shape(self) -> Tuple[int, int, int]:
        """Get the expected feature shape"""
        return (self.n_mels, self.target_length, 1)
    
    def get_feature_dimension(self):
        """Return the feature dimension for mel-spectrogram"""
        # For CNN input: (n_mels, time_steps, channels)
        return self.get_feature_shape()
    
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
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'target_length': self.target_length,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'feature_shape': self.get_feature_shape()
        }