from bitmap import signal_to_bitmap, plot_bitmap
import numpy as np
import matplotlib.pyplot as plt
import pickle

data_path = 'data/'
fs = 1000
t = np.linspace(0, 1, fs)
frequencies = np.linspace(2, 20, 100)
noise_levels = np.linspace(1/50, 0.5, 20)


dataset = {
    'frequencies': frequencies,
    'noise_levels': noise_levels,
    'time': t,
    'signals': {
        'original': [],
        'noisy': []
    },
    'bitmaps': {
        'original': [],
        'noisy': []
    }
}


for f in frequencies:
    signal = np.sin(2 * np.pi * f * t)
    bitmap = signal_to_bitmap(signal)

    dataset['signals']['original'].append(signal)
    dataset['bitmaps']['original'].append(bitmap)

    for noise in noise_levels:
        noisy_signal = signal + np.random.normal(0, noise, fs)
        noisy_bitmap = signal_to_bitmap(noisy_signal)
        intersection = np.logical_and(bitmap, noisy_bitmap).sum()
        union = np.logical_or(bitmap, noisy_bitmap).sum()
        iou = intersection / union if union != 0 else 0
        print(f'Frequency: {f}, Noise: {noise}, IoU: {iou}')

        dataset['signals']['noisy'].append(noisy_signal)
        dataset['bitmaps']['noisy'].append(noisy_bitmap)

    # plot_bitmap(signal,t, bitmap)


with open(data_path + 'dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
