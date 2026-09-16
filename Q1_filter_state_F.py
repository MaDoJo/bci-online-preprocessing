"""
Compare difference between passing filter state (zi)
and restarting the filter at 0 each chunk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from src.filters import OnlineBandpass, Stateless

FS = 250
N_CHANNELS = 1

signal = np.random.randn(FS * 5, N_CHANNELS)
chunk_size = FS

stateful_filt = OnlineBandpass(FS, 8, 30, N_CHANNELS)
stateful = [stateful_filt.process([s])[0] for s in signal[:, 0]]

stateless_filt = Stateless(FS, 8, 30, N_CHANNELS)
stateless = [stateless_filt.process([s])[0] for s in signal[:, 0]]

plt.plot(stateless, label="stateless (causal)")
plt.plot(stateful, label="stateful (causal)")
plt.legend()
plt.title("Stateful vs stateless Filtering")
plt.show()
