import numpy as np
import wave
from pathlib import Path
from typing import Tuple, Optional
import scipy.signal

class AudioProcessor:
    """Handle audio loading, preprocessing, and normalization"""
    
    def __init__(self, target_sr: int = 8000):
        self.target_sr = target_sr
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate"""
        try:
            # Try to load with wave for WAV files
            with wave.open(filepath, 'rb') as wav_file:
                sr = wav_file.getframerate()
                frames = wav_file.readframes(-1)
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Resample if needed
                if sr != self.target_sr:
                    audio = scipy.signal.resample(audio, int(len(audio) * self.target_sr / sr))
                
                return audio, self.target_sr
        except Exception as e:
            # If file doesn't exist (demo mode), return synthetic audio
            print(f"Warning: Could not load {filepath}, generating synthetic audio")
            return self._generate_synthetic_audio(), self.target_sr
    
    def _generate_synthetic_audio(self) -> np.ndarray:
        """Generate synthetic audio for demonstration purposes"""
        # Create a simple synthetic audio signal
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.target_sr * duration))
        
        # Create a simple tone with some harmonics
        frequency = 440  # A4 note
        audio = (np.sin(2 * np.pi * frequency * t) * 0.3 +
                np.sin(2 * np.pi * frequency * 2 * t) * 0.1 +
                np.sin(2 * np.pi * frequency * 3 * t) * 0.05)
        
        # Add some noise
        noise = np.random.normal(0, 0.02, audio.shape)
        audio += noise
        
        return audio.astype(np.float32)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Remove leading and trailing silence"""
        # Simple energy-based silence trimming
        # Calculate energy threshold
        energy = np.abs(audio)
        threshold = np.max(energy) * (10 ** (-top_db / 20))
        
        # Find non-silent regions
        above_threshold = energy > threshold
        if not np.any(above_threshold):
            return audio
        
        # Find first and last non-silent samples
        first = np.argmax(above_threshold)
        last = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1
        
        return audio[first:last+1]
    
    def preprocess_audio(self, filepath: str) -> np.ndarray:
        """Complete audio preprocessing pipeline"""
        # Load audio
        audio, sr = self.load_audio(filepath)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Trim silence
        audio = self.trim_silence(audio)
        
        # Ensure minimum length (pad if too short)
        min_length = int(0.1 * self.target_sr)  # 100ms minimum
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        return audio
    
    def save_audio(self, audio: np.ndarray, filepath: str):
        """Save audio to file"""
        # Convert to int16 and save with wave
        audio_int = (audio * 32767).astype(np.int16)
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.target_sr)
            wav_file.writeframes(audio_int.tobytes())
