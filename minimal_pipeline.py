"""
Minimal real-time preprocessing pipeline
"""

from pylsl import StreamInlet, resolve_byprop
from src.filters import OnlineBandpass
from src.windowing import SlidingWindow
from scipy.signal import butter, filtfilt
import numpy as np

FS = 250
N_CHANNELS = 8
DELAY = 15
WINDOW_SIZE = DELAY * 2 + 1 

streams = resolve_byprop("type", "EEG")
inlet = StreamInlet(streams[0])
b, a = butter(4, [8, 30], fs=FS, btype="band")

filt = OnlineBandpass(fs=FS, low=8, high=30, n_channels=N_CHANNELS)
window = SlidingWindow(size=FS, step=FS // 4)

while True:
    chunk, timestamps = inlet.pull_chunk()
    if not chunk:
            continue
    
    # Process order: Window -> Filter
    for idx, sample in enumerate(chunk):
        ts = timestamps[idx]
        
        # Feed multi-channel sample to the window
        win = window.update(sample)
        if win is not None:
            print("Window ready:", win.shape)
            # Transpose so time is on the horizontal axis
            window_data = np.array(win).T 
            
            # Apply non-causal filter across the time axis
            filtered_window = filtfilt(b, a, window_data, axis=1)
            center_sample = filtered_window[:, DELAY]
    