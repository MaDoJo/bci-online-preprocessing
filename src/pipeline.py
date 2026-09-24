"""
Minimal real-time preprocessing pipeline
"""

from pylsl import StreamInlet, resolve_byprop
from src.filters import OnlineBandpass
from src.windowing import SlidingWindow

FS = 250
N_CHANNELS = 8

streams = resolve_byprop("type", "EEG")
inlet = StreamInlet(streams[0])

filt = OnlineBandpass(fs=FS, low=8, high=30, n_channels=N_CHANNELS)
window = SlidingWindow(size=FS, step=FS // 4)

while True:
    sample, _ = inlet.pull_sample()
    sample = filt.process(sample)
    win = window.update(sample)

    if win is not None:
        print("Window ready:", win.shape)
