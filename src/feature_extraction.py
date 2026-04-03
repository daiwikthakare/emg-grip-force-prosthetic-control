import numpy as np

def extract_features(window):
    mav = np.mean(np.abs(window), axis=0)
    rms = np.sqrt(np.mean(window**2, axis=0))
    zc = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)
    ssc = np.sum(np.diff(np.sign(np.diff(window, axis=0)), axis=0) != 0, axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    return np.concatenate([mav, rms, zc, ssc, wl])

def segment_signal(signal, window_size=200, overlap=0.5):
    step = int(window_size * (1 - overlap))
    segments = []
    for i in range(0, len(signal) - window_size, step):
        segments.append(signal[i:i+window_size])
    return segments
