import numpy as np
import matplotlib.pyplot as plt

def signal_to_bitmap(signal, resolution=100, padding=20, threshold=None):
    """
    Convert a 1D signal to a bitmap.

    Parameters:
    - signal (array_like): The 1D signal to convert.
    - resolution (int): Number of quantization levels for the bitmap. Default is 100.
    - threshold (float or None): Threshold value for binarizing the bitmap. If None, no binarization is performed. Default is None.

    Returns:
    - bitmap (ndarray): The bitmap representation of the signal.
    """
    # Normalize signal to [0, 1] range
    normalized_signal = (signal - np.min(signal)) / \
        (np.max(signal) - np.min(signal))

    # Quantize the normalized signal
    quantized_signal = np.round(
        normalized_signal * (resolution - 1)).astype(int)

    # Create bitmap
    bitmap = np.zeros((resolution, len(signal)), dtype=int)
    for i in range(len(signal)):
        bitmap[resolution-quantized_signal[i]-1, i] = 1

    # Binarize bitmap based on threshold
    if threshold is not None:
        bitmap[bitmap <= threshold] = 0
        bitmap[bitmap > threshold] = 1

    # Add padding to the bitmap
    if padding > 0:
        bitmap = np.pad(bitmap, ((padding, padding), (0, 0)), mode='constant')

    return bitmap

def plot_bitmap(signal,time,bitmap):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.imshow(bitmap, cmap='gray', aspect='auto')
    plt.title('Signal Bitmap')
    plt.xlabel('Time')
    plt.ylabel('Samples')

    plt.tight_layout()
    plt.show()