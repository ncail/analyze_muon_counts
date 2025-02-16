# Imports.
import datetime
import json
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import pywt


''' ***************************************** CONFIG ******************************************** '''
# Config mode (of plotting results): 0 - raw data, 1 - fft, 2 - wavelet
mode = 0

# Get input and output paths.
data_filepath = 'preprocessed_data/preprocessed_1M-intervals_20241004_120111_manually_trimmed.csv'

# Config signal smoothing.
gaussian_smoothing_sigma = 0

# Config fft post-processing.
low_pass_filter = 0  # Cutoff frequency.

''' *************************************** PROCESSING ****************************************** '''
# Get timestamp for filenames.
current_timestamp = datetime.datetime.now().strftime('%HH%MM%SS_%mm%dd%YYYY')

# Read file into dataframe. File should be muon detector CSV file.
df = pd.read_csv(data_filepath, parse_dates=[0])

# Delete invalid rows, convert data to pandas standards:
# timestamp column to datetime, and count column to numeric.
column_names = df.columns
timestamp_col = column_names[0]
count_col = column_names[1]

# Convert the event counts to a numpy array.
event_counts_raw = df[count_col]

if gaussian_smoothing_sigma:
    event_counts_raw = gaussian_filter1d(event_counts_raw, sigma=gaussian_smoothing_sigma)

# Remove expected constant signal/background noise.
background_level = np.mean(event_counts_raw)
event_counts_detrended = event_counts_raw - background_level

# Apply Fourier Transform on both arrays.
# fft_values_raw = rfft(event_counts_raw)
fft_values_detrended = rfft(event_counts_detrended)

# Get the frequencies associated with the FFT results.
n = len(event_counts_raw)
time_interval = df.loc[1, timestamp_col] - df.loc[0, timestamp_col]  # Assumes data has been preprocessed into equal
# time samples.
frequencies_domain = rfftfreq(n, d=int(time_interval.total_seconds()))

if low_pass_filter:
    fft_values_detrended[frequencies_domain > low_pass_filter] = 0

# Get magnitudes of frequency components: shows the strength of each frequency component.
# magnitudes_raw = np.abs(fft_values_raw)
magnitudes_detrended = np.abs(fft_values_detrended)

# Get strongest frequency component.
max_mag_pos = magnitudes_detrended.argmax()
max_mag_frequency = frequencies_domain[max_mag_pos]

# Wavelet analysis.
scales = np.arange(1, len(df[count_col])/2)
coeffs, frequencies_wt = pywt.cwt(df[count_col], scales, 'cmor')

''' **************************************** PLOT RESULTS *************************************** '''
# Get timestamps for plotting raw data and wavelet transform.
time_series = df[timestamp_col].tolist()


# Plot raw data (and smoothed data) start.
# To identify transient events, which may correspond to solar flares, etc.
if mode == 0:
    # Plot mean value (background level).
    plt.plot(time_series, event_counts_raw, label="Data")
    plt.plot(time_series, np.full(len(time_series), background_level), color="red", label="Average")
    plt.suptitle("Raw event counts")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.ylabel(f"Event counts (per {int(time_interval.total_seconds())} sec)")
    plt.legend()
    plt.grid(True)

    if gaussian_smoothing_sigma:
        plt.title(f'Gaussian-smoothed: sigma={gaussian_smoothing_sigma}')

    plt.savefig(f'results/{current_timestamp}_raw_data_plot.png')

    plt.show()
# Plot raw data end.


# Plot ffts start.
if mode == 1:
    plt.plot(frequencies_domain, magnitudes_detrended, color='orange')

    # Draw and label frequency of max power in per hours.
    max_mag_freq_in_hours = (1 / max_mag_frequency) / 3600
    plt.axvline(x=max_mag_frequency, color='m', linestyle='--',
                label=f'Maximum magnitude: {round(max_mag_freq_in_hours, 2)}-hour frequency')

    plt.title("FFT of Detrended Data")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'results/{current_timestamp}_fft_plot.png')

    plt.show()
# Plot ffts end.


# Plot wavelet transform start.
if mode == 2:
    # Plot cone of influence.

    # Convert scales to periods in hours.

    plt.imshow(np.abs(coeffs), aspect='auto', extent=[min(time_series), max(time_series), scales.max(), scales.min()])
    plt.colorbar(label='Power')
    plt.xlabel("Time")
    plt.ylabel("Scale")
    plt.title("Wavelet Transform of Muon Counts")

    plt.savefig(f'results/{current_timestamp}_wavelet_plot.png')

    plt.show()
# Plot wavelet transform end.


# Log current configs, write to file.
log_configs = {
    "mode [raw data, fft, wavelet]": mode,
    "data filepath": data_filepath,
    "sampling interval": time_interval,
    "gaussian smoothing sigma parameter": gaussian_smoothing_sigma,
    "low pass filter cutoff frequency": low_pass_filter
}
with open(f"results/{current_timestamp}_log.txt", "w") as f:
    json.dump(log_configs, f, indent=4)



