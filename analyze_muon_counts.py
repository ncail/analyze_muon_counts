# Imports.
import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import pywt


# Program start.
''' ***************************************** CONFIG ******************************************** '''
# Config mode (of plotting results): 0 - raw data, 1 - fft, 2 - wavelet
mode = 2

# Get input and output paths.
data_filepath = 'preprocessed_data/preprocessed_1H-intervals_20241004_120111_manually_trimmed.csv'


''' *************************************** PROCESSING ****************************************** '''
# Read file into dataframe. File should be muon detector CSV file.
df = pd.read_csv(data_filepath, parse_dates=[0])

# Delete invalid rows, convert data to pandas standards:
# timestamp column to datetime, and count column to numeric.
column_names = df.columns
timestamp_col = column_names[0]
count_col = column_names[1]

# Convert the event counts to a numpy array.
event_counts_raw = df[count_col]

# Remove expected constant signal/background noise.
background_level = np.mean(event_counts_raw)
event_counts_detrended = event_counts_raw - background_level

# Apply Fourier Transform on both arrays.
fft_values_raw = fft(event_counts_raw)
fft_values_detrended = fft(event_counts_detrended)

# Get the frequencies associated with the FFT results.
n = len(event_counts_raw)
time_interval = df.loc[1, timestamp_col] - df.loc[0, timestamp_col]  # Assumes data has been preprocessed into equal
# time samples.
frequencies_fft = np.fft.fftfreq(n, d=int(time_interval.total_seconds()))

# Get magnitudes of frequency components: shows the strength of each frequency component.
magnitude_raw = np.abs(fft_values_raw)
magnitude_detrended = np.abs(fft_values_detrended)

# Wavelet analysis.
scales = np.arange(1, len(df[count_col])/2)
coeffs, frequencies_wt = pywt.cwt(df[count_col], scales, 'cmor')

''' **************************************** PLOT RESULTS *************************************** '''
# Get timestamps for plotting raw data.
time_series = df[timestamp_col].tolist()

# Only take the positive frequencies (frequencies > 0) for plotting fft.
positive_frequencies = frequencies_fft[:n // 2]
positive_magnitude_raw = magnitude_raw[:n // 2]
positive_magnitude_detrended = magnitude_detrended[:n // 2]


# Plot raw data start.
if mode == 0:
    # To identify non-recurring events, which may correspond to solar flares, etc.
    # Plot mean value (background level).
    plt.plot(time_series, event_counts_raw, label="Data")
    plt.plot(time_series, np.full(len(time_series), background_level), color="red", label="Average")
    plt.title("Raw event counts")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.ylabel(f"Event counts (per {int(time_interval.total_seconds())} sec)")
    plt.legend()
    plt.grid(True)
    plt.show()
# Plot raw data end.


# Plot ffts start.
if mode == 1:
    # Plot raw and detrended data FFT magnitude.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(positive_frequencies, positive_magnitude_raw, color='blue')
    plt.title("FFT of Raw Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(positive_frequencies, positive_magnitude_detrended, color='orange')

    # Draw and label frequency of max power in per hours.
    max_mag_pos = positive_magnitude_detrended.argmax()
    max_mag_frequency = positive_frequencies[max_mag_pos]
    max_mag_freq_in_hours = (1 / max_mag_frequency) / 3600
    plt.axvline(x=max_mag_frequency, color='m', linestyle='--',
                label=f'Maximum magnitude: {round(max_mag_freq_in_hours, 2)}-hour frequency')

    plt.title("FFT of Detrended Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# Plot ffts end.


# Plot wavelet transform start.
if mode == 2:
    plt.imshow(np.abs(coeffs), aspect='auto', extent=[min(time_series), max(time_series), scales.max(), scales.min()])
    plt.colorbar(label='Power')
    plt.xlabel("Time")
    plt.ylabel("Scale")
    plt.title("Wavelet Transform of Muon Counts")
    plt.show()
# Plot wavelet transform end.






