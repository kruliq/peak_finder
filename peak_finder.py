from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.io.wavfile as wav
from scipy.ndimage import uniform_filter1d
import pandas as pd
import os

sample_rate, data = wav.read('skrzypce.wav')


fft_data = np.fft.fft(data, 16384)
freq_bins = np.fft.fftfreq(16384, 1 / 16000)
positive_freq_indices = np.where(freq_bins >= 0)
positive_freq_bins = freq_bins[positive_freq_indices]
positive_fft_data = np.abs(fft_data[positive_freq_indices])

# Apply smoothing 
positive_fft_data_smoothed = uniform_filter1d(positive_fft_data, size=8)

# Convert the magnitude spectrum to dBFS
positive_fft_data_dbfs = 20 * np.log10(positive_fft_data_smoothed / np.max(positive_fft_data_smoothed))

# Find peaks
peaks, properties = find_peaks(positive_fft_data_dbfs, height=-60, distance=60)

# Filter peaks that are at least 10 dB above the surrounding noise floor
prominent_peaks = []
for i, peak in enumerate(peaks):
    start = max(0, peak - 20)
    end = min(len(positive_fft_data_dbfs), peak + 20)
    surrounding_noise_floor = np.mean(positive_fft_data_dbfs[start:end])
    if properties['peak_heights'][i] - surrounding_noise_floor >= 10:
        prominent_peaks.append(peak)

# Plot
plt.plot(positive_freq_bins, positive_fft_data_dbfs)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dBFS)')
plt.title('FFT of Resampled Data (Positive Frequencies)')

# Label peaks
for peak in prominent_peaks:
    plt.annotate(f'{positive_freq_bins[peak]:.1f} Hz', 
                 (positive_freq_bins[peak], positive_fft_data_dbfs[peak]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

plt.show()

export_data = {
    'Frequency (Hz)': positive_freq_bins[prominent_peaks],
    'Magnitude (dBFS)': positive_fft_data_dbfs[prominent_peaks]
}

# Export to Excel
df = pd.DataFrame(export_data)

def get_unique_filename(base_filename, extension):
    counter = 1
    filename = f"{base_filename}.{extension}"
    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}.{extension}"
        counter += 1
    return filename

unique_filename = get_unique_filename('peaks_data', 'xlsx')

df.to_excel(unique_filename, index=False)