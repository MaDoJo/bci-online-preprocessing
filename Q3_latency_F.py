from src.filters import OnlineBandpass
import time
import numpy as np

# Measure processing time: one sample at a time

FS = 250
N_CHANNELS = 1

# Use the randomly generated signal
signal = np.random.randn(FS * 5, N_CHANNELS)

# Initiate filter
single_filter = OnlineBandpass(FS, 8, 30, N_CHANNELS)

single_times = []

# Use high-resolution timer, time each sample in signal.
for sample in signal:
    start = time.perf_counter()

    filtered_sample = single_filter.process(sample)

    end = time.perf_counter()

    single_times.append(end - start)

# time.perf_counter() gives seconds, so convert to ms
single_times_ms = np.array(single_times) * 1000

print("Latency of causal filter, one sample at a time")
print(f"Average time per sample: {np.mean(single_times_ms):.4f} ms")
print(f"Maximum time per sample: {np.max(single_times_ms):.4f} ms")


# Chunk version
chunk_filter = OnlineBandpass(FS, 8, 30, N_CHANNELS)
CHUNK_SIZE = 25
chunk_times = []
results = []

# go through the signal from 0 to len(signal) in jumps of chunk size
for i in range(0, len(signal), CHUNK_SIZE):
    # update chunk
    chunk = signal[i:(i + CHUNK_SIZE)]

    # Use equally sized chunks for a fair comparison
    # e.g. if chunk size is 3, and len(signal) is 10
    # then the chunks are (0,1,2), (3,4,5), (6,7,8), (9)
    # so the last chunk that runs is only len = 1.
    if len(chunk) != CHUNK_SIZE:
        continue

    # start timer
    start = time.perf_counter()
    # perform filtering
    chunk_filter.process_chunk(chunk)
    # end timer
    end = time.perf_counter()

    chunk_times.append(end - start)


# convert seconds to ms & get mean chunk time
chunk_times_ms = np.array(chunk_times) * 1000
mean_chunk_time_ms = np.mean(chunk_times) * 1000
mean_time_per_sample_ms = mean_chunk_time_ms / CHUNK_SIZE


print("Latency of causal filter, per chunk")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Average time per chunk: {mean_chunk_time_ms:.4f} ms")
print(f"Maximum time per chunk: {np.max(chunk_times_ms):.4f} ms")
print(f"Average time per sample in chunk: " 
      f"{(mean_chunk_time_ms/CHUNK_SIZE):.4f} ms")
print(f"Maximum time per chunk divided by chunk size: "
      f"{(np.max(chunk_times_ms)/CHUNK_SIZE):.4f} ms")
