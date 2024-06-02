import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from bitmap import signal_to_bitmap


noise = np.linspace(0, 1, 100)
correlation_original = []
correlation_1D = []
miou = []
ssim_index = []

# plt.figure(figsize=(10, 6))
# plt.plot(np.random.normal(0, 0.4, 1000))
# plt.title('Noisy Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()


for noise_amplitude in noise:

    num_samples = 1000  # Number of samples
    frequency = 2  # Frequency of the sine wave
    amplitude = 1  # Amplitude of the sine wave

    # Generate time axis
    t = np.linspace(0, 1, num_samples)

    # Generate pure sine signal
    pure_sine_signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Generate bitmap of the pure sine signal
    pure_sine_bitmap = signal_to_bitmap(pure_sine_signal)

    # Amplitude of the noise

    # Add noise to the pure sine signal
    noisy_signal = pure_sine_signal + np.random.normal(0, noise_amplitude, num_samples)

    correlation_original.append(np.corrcoef(
        pure_sine_signal, noisy_signal)[0, 1])

    # Generate bitmap of the noisy signal
    noisy_bitmap = signal_to_bitmap(noisy_signal)

    # Calculate correlation between the pure sine bitmap and the noisy bitmap
    correlation_1D.append(np.corrcoef(
        pure_sine_bitmap.flatten(), noisy_bitmap.flatten())[0, 1])

    intersection = np.logical_and(pure_sine_bitmap, noisy_bitmap).sum()
    union = np.logical_or(pure_sine_bitmap, noisy_bitmap).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0

    # For binary classification, MIoU is the same as IoU
    miou.append(iou)

    # print("Mean Intersection over Union (MIoU):", miou)

    ssim_index.append(ssim(pure_sine_bitmap, noisy_bitmap,
                           data_range=noisy_bitmap.max() - noisy_bitmap.min()))

    # print("Structural Similarity Index (SSI):", ssim_index)


noisy_signal = pure_sine_signal + np.random.normal(0, 0.1, num_samples)
noisy_bitmap = signal_to_bitmap(noisy_signal)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.imshow(pure_sine_bitmap, cmap='gray', aspect='auto')
plt.title('Original Signal Bitmap')
plt.xlabel('Time')
plt.ylabel('Samples')


plt.subplot(3, 1, 2)
plt.imshow(noisy_bitmap, cmap='gray', aspect='auto')
plt.title('Noisy Signal Bitmap')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.subplot(3, 1, 3)
plt.plot(-noisy_signal)
plt.title('Noisy Signal')
plt.xlabel('Time')
plt.ylabel('Samples')

plt.suptitle('$\sigma = 0.1$, q = 100')
plt.tight_layout()



plt.figure(figsize=(10, 6))
plt.plot(noise, correlation_original,
         label='1D Correlation')

# Plot Correlation vs. Noise Factor
plt.plot(noise, correlation_1D,  label='Correlation')

# Plot MIoU vs. Noise Factor
plt.plot(noise, miou,  label='mIoU')

# Plot SSI vs. Noise Factor
plt.plot(noise, ssim_index, label='SSI')

plt.title('Similarity Metrics with Gaussian Noise')
plt.xlabel('Noise Level (Standard Deviation $\sigma$)')
plt.ylabel('Metric Value')
plt.grid(True)
plt.legend()  # Add a legend to distinguish the plots

plt.show()
