# Imports.
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
from scipy.fft import fft
from scipy.signal import welch
import matplotlib.pyplot as plt


# Program start.
''' *************************************** CONFIG ****************************************** '''
# Get input and output paths.
data_filename = ''
data_filepath = f''
output_path = 'results'

current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_filename = f'results_{current_timestamp}'

# Set parameter for Welch's method for getting PSD.
welch_segment_length = config['processing']['welch_segment_length']

''' *************************************** PROCESSING ****************************************** '''
# Log processing info of current run.


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
frequencies = np.fft.fftfreq(n, d=int(time_interval.total_seconds()))

# Get magnitudes of frequency components: shows the strength of each frequency component.
magnitude_raw = np.abs(fft_values_raw)
magnitude_detrended = np.abs(fft_values_detrended)

# Wavelet analysis.

''' ****************************************** RESULTS ****************************************** '''
# Get timestamps for plotting raw data.
time_series = df[timestamp_col].tolist()

# Only take the positive frequencies (frequencies > 0) for plotting fft.
positive_frequencies = frequencies[:n // 2]
positive_magnitude_raw = magnitude_raw[:n // 2]
positive_magnitude_detrended = magnitude_detrended[:n // 2]


# Plot raw data start.
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

# Program end.







