import numpy as np
import pylsl
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from src.inlet import create_inlet
from src.buffer import CircularBuffer
from src.windowing import SlidingWindow

FS = 250
N_CHANNELS = 8
BUFFER_SEC = 2
DELAY = 15
WINDOW_SIZE = DELAY * 2 + 1  

inlet = create_inlet()
buffer = CircularBuffer(FS * BUFFER_SEC, N_CHANNELS)
window_builder = SlidingWindow(size=WINDOW_SIZE, step=1)
b, a = butter(4, [8, 30], fs=FS, btype="band")

fig, ax = plt.subplots()

while True:
    # pull_chunk retrieves all available samples in the LSL buffer at once
    chunk, timestamps = inlet.pull_chunk()

    if not chunk:
        continue

    # Process order: Window -> Filter
    for idx, sample in enumerate(chunk):
        ts = timestamps[idx]
        
        # Feed multi-channel sample to the window
        window = window_builder.update(sample)
        
        if window is not None:
            # Transpose so time is on the horizontal axis
            window_data = np.array(window).T 
            
            # Apply non-causal filter across the time axis
            print(window_data)
            filtered_window = filtfilt(b, a, window_data, axis=1)
            center_sample = filtered_window[:, DELAY]
            
            # Multiply channel index by -50 mV for channel visual spacing
            for ch_idx in range(N_CHANNELS):
                center_sample[ch_idx] += (ch_idx * -50.0)
                
            buffer.append(center_sample)
            

    data = buffer.get()
    if data is not None and len(data) > 0:
        ax.clear()
        ax.plot(data)
        ax.set_title("Live Multichannel EEG (Quasi-Causal)")
        plt.pause(0.01)
