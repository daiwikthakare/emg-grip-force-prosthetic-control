import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(signal, fs=2000, low=20, high=450, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal, axis=0)

def notch_filter(signal, fs=2000, freq=50):
    b, a = iirnotch(freq/(fs/2), Q=30)
    return filtfilt(b, a, signal, axis=0)

def preprocess(signal):
    signal = bandpass_filter(signal)
    signal = notch_filter(signal)
    signal = np.abs(signal)  # rectification
    return signal
