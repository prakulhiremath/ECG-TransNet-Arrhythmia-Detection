import wfdb
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

class MITBIHLoader:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        # AAMI Class Mapping
        self.category_map = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,          # N: Normal
            'A': 1, 'a': 1, 'J': 1, 'S': 1,                 # S: Supraventricular
            'V': 2, 'E': 2,                                 # V: Ventricular
            'F': 3,                                         # F: Fusion
            '/': 4, 'f': 4, 'Q': 4                          # Q: Unknown/Paced
        }

    def apply_filter(self, data):
        """Remove baseline wander and high-frequency noise."""
        # High-pass filter to remove baseline wander (0.5 Hz)
        sos = signal.butter(4, 0.5, 'hp', fs=360, output='sos')
        filtered = signal.sosfilt(sos, data)
        # Low-pass filter (35 Hz)
        sos = signal.butter(4, 35, 'lp', fs=360, output='sos')
        return signal.sosfilt(sos, filtered)

    def segment_beats(self, record_path):
        """Load record and segment into 1-second windows around R-peaks."""
        # Read the record (channel 0 usually Lead II)
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')
        
        sig = self.apply_filter(record.p_signal[:, 0])
        
        beats = []
        labels = []
        
        # Window size: 0.7s (250 samples at 250Hz approx)
        # We take 100 samples before R-peak and 150 after
        for i, (idx, sym) in enumerate(zip(ann.sample, ann.symbol)):
            if sym in self.category_map:
                if idx > 100 and idx < len(sig) - 150:
                    beat = sig[idx-100 : idx+150]
                    # Resample to the target rate if necessary
                    if record.fs != self.sampling_rate:
                        beat = signal.resample(beat, 250)
                    
                    beats.append(beat)
                    labels.append(self.category_map[sym])
        
        return np.array(beats), np.array(labels)

    def load_dataset(self, record_ids):
        """Processes a list of records and returns combined X and y."""
        all_beats = []
        all_labels = []
        
        for rid in record_ids:
            print(f"Processing Record: {rid}...")
            b, l = self.segment_beats(f'data/mit-bih-arrhythmia/{rid}')
            all_beats.append(b)
            all_labels.append(l)
            
        X = np.concatenate(all_beats, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Standardize (Mean=0, Std=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Reshape for 1D-CNN (Batch, Channels, Length)
        X = X.reshape(-1, 1, 250)
        return X, y

if __name__ == "__main__":
    loader = MITBIHLoader()
    X, y = loader.load_dataset(['100'])
    print("Dataset loaded successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

