"""
LSL EEG inlet wrapper
"""

from pylsl import StreamInlet, resolve_byprop

def create_inlet(stream_type="EEG"):
    print("Looking for EEG stream...")
    streams = resolve_byprop(
        "type",
        "EEG",
        timeout=5
    )
    inlet = StreamInlet(streams[0])
    return inlet

