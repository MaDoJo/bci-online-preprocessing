"""
Compare offline and online filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, group_delay
from src.filters import OnlineBandpass

FS = 250
N_CHANNELS = 1

signal = np.random.randn(FS * 5, N_CHANNELS)

b, a = butter(4, [8, 30], fs=FS, btype="band")
offline = filtfilt(b, a, signal[:, 0])

online_filt = OnlineBandpass(FS, 8, 30, N_CHANNELS)
online = [online_filt.process([s])[0] for s in signal[:, 0]]

# Plot 1: like in examples folder
plt.plot(offline, label="offline (filtfilt)")
plt.plot(online, label="online (causal)")
plt.legend()
plt.title("Offline vs Online Filtering")
plt.show()

# Plot 2: only the first second
plt.plot(offline[:FS], label="offline (filtfilt)")
plt.plot(online[:FS], label="online (causal)")
plt.legend()
plt.title("Offline vs Online Filtering: first 1 sec frame")
plt.show()

# Plot 3:
# A version with how it might look if the
# phase delay of online filtering was smaller
# to see if the signals would overlap then

N = 250
DELAY = 15

# First 1s
offline_frame = offline[:N]
online_frame = np.array(online[:N])     # make it an array, to do the shift

# Shift causal signal 15 samples earlier
# Make array like the online_frame (np.full_like)
online_shifted = np.full_like(online_frame, np.nan) 
# the new array is the old array starting from the delay time
online_shifted[:-DELAY] = online_frame[DELAY:]

plt.figure(figsize=(10, 5))

plt.plot(offline_frame, label="offline (filtfilt)")
plt.plot(online_shifted, label="online (causal, shifted 15 samples earlier)")

plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Offline vs Online Filtering, with online filtering shifted -15 samples")
plt.legend()
plt.show()



# Plot 4: visualizing phase shift and time delay (ms)
LOW = 8
HIGH = 30

# Convert samples -> milliseconds
delay_ms = delay_samples / FS * 1000

