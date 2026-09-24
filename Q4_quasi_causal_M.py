"""
Compare offline, online causal, and quasi-causal filtering using sliding windows
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from src.filters import OnlineBandpass
from src.windowing import SlidingWindow  

FS = 250
N_CHANNELS = 1

signal = np.random.randn(FS * 5, N_CHANNELS)

# Causal online filtering
causal_filt = OnlineBandpass(FS, 8, 30, N_CHANNELS)
online_causal = [causal_filt.process([s])[0] for s in signal[:, 0]]

# Quasi-causal filtering
DELAY = 15  
WINDOW_SIZE = DELAY * 2 + 1  # 31 samples

window_builder = SlidingWindow(size=WINDOW_SIZE, step=1)
online_quasi = []

b, a = butter(4, [8, 30], fs=FS, btype="band")

for sample in signal[:, 0]:
    # Update sliding window
    window = window_builder.update([sample])
    
    if window is not None:
        window_data = window.squeeze()
        # non-causal filter
        filtered_window = filtfilt(b, a, window_data)
        
        # center sample is the quasi-causal output at t - DELAY
        center_sample = filtered_window[DELAY]
        online_quasi.append(center_sample)
    else:
        # delay before first full window is ready
        online_quasi.append(np.nan)


online_quasi = np.array(online_quasi)

# Plot: first 1 second frame
N_plot = FS
plt.figure(figsize=(10, 5))

plt.plot(online_causal[:N_plot], label="online (causal)", linewidth=1.5, color="orange")
plt.plot(online_quasi[:N_plot], label=f"quasi-causal (windowed filtfilt, delay={DELAY})", linestyle="--", color="red", linewidth=1.5)
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Online Causal vs Quasi-causal Filtering")
plt.legend()
plt.grid(True)
plt.show()