import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List

class FSIDDataset:
    """Handler for Free Spoken Digit Dataset (FSDD)"""
    
    def __init__(self, data_dir: str = "./fsdd_data"):
        self.data_dir = Path(data_dir)
        self.metadata = None
        self.splits = None
        
    def load_dataset(self):
        """Load or download FSDD dataset"""
        try:
            # Try to download and load real FSDD data
            self._download_fsdd_dataset()
        except Exception as e:
            print(f"Could not load real dataset: {e}")
            print("Using synthetic data for demonstration...")
            self._create_demo_metadata()
            
    def _create_demo_metadata(self):
        """Create demo metadata when dataset is not available"""
        # This creates a synthetic dataset structure for development
        metadata_list = []
        
        # Simulate FSDD structure
        speakers = ['jackson', 'nicolas', 'theo', 'yweweler', 'george', 'lucas']
        
        for digit in range(10):
            for speaker in speakers:
                for recording in range(10):  # 10 recordings per speaker per digit
                    filepath = f"demo_{digit}_{speaker}_{recording:02d}.wav"
                    metadata_list.append({
                        'filepath': filepath,
                        'digit': digit,
                        'speaker': speaker,
                        'sample_id': len(metadata_list)
                    })
        
        self.metadata = pd.DataFrame(metadata_list)
        print(f"Created demo metadata with {len(self.metadata)} samples")
        print("Note: Using synthetic data for demonstration. For real performance, use actual audio files.")
    
    def create_speaker_holdout_split(self, test_size: float = 0.2, val_size: float = 0.15, random_state: int = 42):
        """Create train/val/test splits with speaker holdout"""
        if self.metadata is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Get unique speakers
        speakers = self.metadata['speaker'].unique()
        
        # Split speakers
        train_speakers, test_speakers = train_test_split(
            speakers, test_size=test_size, random_state=random_state
        )
        
        train_speakers, val_speakers = train_test_split(
            train_speakers, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Create splits based on speakers
        train_mask = self.metadata['speaker'].isin(train_speakers)
        val_mask = self.metadata['speaker'].isin(val_speakers)
        test_mask = self.metadata['speaker'].isin(test_speakers)
        
        self.splits = {
            'train': self.metadata[train_mask].reset_index(drop=True),
            'val': self.metadata[val_mask].reset_index(drop=True),
            'test': self.metadata[test_mask].reset_index(drop=True)
        }
        
        return self.splits
    
    def get_split_info(self) -> Dict:
        """Get information about the data splits"""
        if self.splits is None:
            return {}
        
        info = {}
        for split_name, split_data in self.splits.items():
            info[split_name] = {
                'num_samples': len(split_data),
                'num_speakers': split_data['speaker'].nunique(),
                'digit_distribution': split_data['digit'].value_counts().to_dict(),
                'speaker_distribution': split_data['speaker'].value_counts().to_dict()
            }
        
        return info
    
    def _download_fsdd_dataset(self):
        """Download real FSDD dataset from GitHub"""
        import urllib.request
        import zipfile
        import tempfile
        
        # Create data directory
        self.data_dir.mkdir(exist_ok=True)
        
        # Check if dataset already exists
        existing_files = list(self.data_dir.glob('*.wav'))
        if len(existing_files) > 50:  # If we have some files already
            print(f"Found {len(existing_files)} existing audio files")
            self._create_metadata_from_files(existing_files)
            return
        
        # FSDD GitHub URL
        url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
        
        print("Downloading FSDD dataset from GitHub...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "fsdd.zip")
            
            # Download the zip file
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the recordings directory
            recordings_dir = None
            for root, dirs, files in os.walk(temp_dir):
                if 'recordings' in dirs:
                    recordings_dir = os.path.join(root, 'recordings')
                    break
            
            if recordings_dir is None:
                raise ValueError("Could not find recordings directory in downloaded data")
            
            # Copy WAV files to our data directory
            wav_files = []
            for wav_file in Path(recordings_dir).glob('*.wav'):
                dest_path = self.data_dir / wav_file.name
                import shutil
                shutil.copy2(wav_file, dest_path)
                wav_files.append(dest_path)
            
            print(f"Downloaded {len(wav_files)} audio files")
            self._create_metadata_from_files(wav_files)
    
    def _create_metadata_from_files(self, audio_files):
        """Create metadata from actual audio files"""
        metadata_list = []
        
        for i, filepath in enumerate(audio_files):
            filename = os.path.basename(filepath)
            
            # Parse filename: digit_speaker_index.wav (e.g., "0_jackson_0.wav")
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                try:
                    digit = int(parts[0])
                    speaker = parts[1]
                    recording_index = parts[2]
                    
                    metadata_list.append({
                        'filepath': str(filepath),
                        'digit': digit,
                        'speaker': speaker,
                        'sample_id': i
                    })
                except (ValueError, IndexError):
                    # Skip files that don't match expected format
                    continue
        
        if len(metadata_list) == 0:
            raise ValueError("No valid audio files found")
        
        self.metadata = pd.DataFrame(metadata_list)
        print(f"Created metadata for {len(self.metadata)} samples")
        print(f"Speakers: {self.metadata['speaker'].unique()}")
        print(f"Digits: {sorted(self.metadata['digit'].unique())}")
