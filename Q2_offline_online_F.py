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
plt.title(
    "Offline vs Online Filtering, with online filtering shifted -15 samples"
    )
plt.legend()
plt.show()


# Plot 4: visualizing phase shift and time delay (ms)

LOW = 8
HIGH = 30

# use freqz function from scipy:
# this tests the frequency response of a filter
frequencies, response = freqz(b, a, fs=FS)

# angle() gives the phase shift in radians
# Convert radians to degrees
phase_degrees = np.angle(response) * 180 / np.pi

# Calculate delay at each frequency
# delay_samples tells us the delay in number of samples
delay_frequencies, delay_samples = group_delay((b, a), fs=FS)

# Convert samples to milliseconds
delay_ms = delay_samples / FS * 1000

# Keep the bandpassed frequencies, 8 to 30 Hz
phase_band = (frequencies >= LOW) & (frequencies <= HIGH)
delay_band = (delay_frequencies >= LOW) & (delay_frequencies <= HIGH)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Top: phase shift
axes[0].plot(frequencies[phase_band], phase_degrees[phase_band])
axes[0].set_title("Phase shift of causal 8–30 Hz bandpass filter")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Phase shift (degrees)")
axes[0].grid()

# Bottom: time delay
axes[1].plot(delay_frequencies[delay_band], delay_ms[delay_band])
axes[1].set_title("Time delay of causal 8–30 Hz bandpass filter")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Group delay (ms)")
axes[1].grid()

plt.tight_layout()
plt.show()
