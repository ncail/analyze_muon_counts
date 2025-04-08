# Imports.
import datetime
import json
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
import pywt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


''' ***************************************** CONFIG ******************************************** '''
# Config mode (of plotting results): 0 - raw data, 1 - fft, 2 - wavelet
mode = 2

# Get input and output paths.
data_filepath = 'preprocessed_data/muon_data/preprocessed_1H-intervals_20250227_132422.csv'
output_path = 'results/spectral_analysis_plots'

# Config signal smoothing.
gaussian_smoothing_sigma = 0

# Config fft post-processing.
low_pass_filter = 1/(6*3600)  # Cutoff frequency.


''' *************************************** PROCESSING ****************************************** '''
# Get timestamp for filenames.
current_timestamp = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')

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
# End if.

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
    # Find the first index where frequency is greater than low_pass_filter.
    idx = np.searchsorted(frequencies_domain, low_pass_filter, side='right')

    # Keep one more than the nearest value.
    frequencies_domain = frequencies_domain[:idx]
    fft_values_detrended = fft_values_detrended[:idx]
# End if.

# Get magnitudes of frequency components: shows the strength of each frequency component.
# magnitudes_raw = np.abs(fft_values_raw)
magnitudes_detrended = np.abs(fft_values_detrended)

# Convert freqs from Hz to cycles per hour.
frequencies_per_hour = frequencies_domain * 3600

# Get indices of top 3 peaks/strongest frequency components.
top5_indices = np.argsort(magnitudes_detrended)[-5:][::-1]  # Top 3 in descending order.

# Get corresponding frequencies and magnitudes.
top5_freqs = frequencies_domain[top5_indices]
top5_mags = magnitudes_detrended[top5_indices]
max_mag_freq_in_hours = 1 / frequencies_per_hour

# Wavelet analysis.
scales = np.arange(1, len(event_counts_detrended)/2)
if mode == 2:
    coeffs, frequencies_wt = pywt.cwt(event_counts_detrended, scales, 'cmor',
                                      sampling_period=time_interval.total_seconds())

# Add mode for testing different wavelets from CWT webpage.


''' **************************************** PLOT RESULTS *************************************** '''
# Get timestamps for plotting raw data and wavelet transform.
time_series = df[timestamp_col].tolist()


# Plot raw data (and smoothed data) start.
# To identify transient events, which may correspond to solar flares, etc.
if mode == 0:
    # Plot mean value (background level).
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, event_counts_raw, label="Data")
    plt.plot(time_series, np.full(len(time_series), background_level), color="red", label="Average")
    plt.suptitle("Detector event counts")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.ylabel(f"Counts (per {int(time_interval.total_seconds())} sec)")
    plt.legend()
    plt.grid(True)

    if gaussian_smoothing_sigma:
        plt.title(f'Gaussian-smoothed: sigma={gaussian_smoothing_sigma}')

    plt.savefig(f'{output_path}/{current_timestamp}_raw_data_plot.png', bbox_inches='tight')

    plt.show()
# Plot raw data end.


# Plot ffts start.
if mode == 1:
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_domain, magnitudes_detrended, color='orange')

    # Draw and label frequency of max power in per hours.
    legend_labels = []
    handles = []
    for ind, peak in enumerate(top5_freqs):
        plt.axvline(x=peak, color='m', linestyle='--')

        # Legend entries.
        dummy_line = mlines.Line2D([], [], color='m', linestyle='-')
        handles.append(dummy_line)
        legend_labels.append(f'Peak {ind + 1}: {max_mag_freq_in_hours[top5_indices[ind]]:.2f}-hour frequency')
    # End for.

    legend = plt.legend(handles=handles, labels=legend_labels, loc='best', fontsize=15)
    legend.get_frame().set_edgecolor('black')

    plt.suptitle("FFT of Detrended Data", fontsize=25)
    plt.xlabel("Frequency (Hz)", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Magnitude", fontsize=20)
    plt.grid(True, alpha=0.75)

    if low_pass_filter:
        plt.title(f"Low pass filter excludes freq>1/{int(1/low_pass_filter)} Hz", fontsize=15)

    plt.savefig(f'{output_path}/{current_timestamp}_fft_plot.png', bbox_inches='tight')

    plt.show()
# Plot ffts end.


# Plot wavelet transform start.
if mode == 2:
    morlet_const = 1  # 0.849

    # Plot cone of influence.
    coi = morlet_const * scales

    # Convert COI values to actual time indices.
    # COI is in terms of the scale, so for each scale, we compute the start and end of the cone.
    coi_start = time_series[0] + pd.to_timedelta(coi * time_interval, unit='s')  # Start of COI at each scale.
    coi_end = time_series[-1] - pd.to_timedelta(coi * time_interval, unit='s')  # End of COI at each scale.

    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(time_series, frequencies_wt, np.abs(coeffs))
    ax.set_xlabel("Date Time", fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)", fontsize=15)
    ax.set_title("Muon Count Wavelet Transform", fontsize=25)
    fig.colorbar(pcm, ax=ax)

    legend_labels = []
    handles = []
    for ind, peak in enumerate(top5_freqs):
        plt.axhline(y=peak, color='m', linestyle='--')

        # Legend entries.
        dummy_line = mlines.Line2D([], [], color='m', linestyle='-')
        handles.append(dummy_line)
        legend_labels.append(f'Peak {ind + 1} from FFT: {max_mag_freq_in_hours[top5_indices[ind]]:.2f}-hour frequency')
    # End for.
    legend_labels.append('COI')
    legend = plt.legend(handles=handles, labels=legend_labels, loc='best', fontsize=12)
    legend.get_frame().set_edgecolor('black')

    # Plot COI.
    plt.plot(coi_start, frequencies_wt, 'w--')
    plt.plot(coi_end, frequencies_wt, 'w--')

    plt.savefig(f'{output_path}/{current_timestamp}_wavelet_plot.png', bbox_inches='tight')

    plt.show()
# Plot wavelet transform end.


''' ***************************************** LOG CONFIGS *************************************** '''
# Log current configs, write to file.
log_configs = {
    "mode [raw data, fft, wavelet]": mode,
    "data filepath": data_filepath,
    "sampling interval": str(time_interval),
    "gaussian smoothing sigma parameter": gaussian_smoothing_sigma,
    "low pass filter cutoff frequency": f'1/{int(1/low_pass_filter)}' if low_pass_filter else 0,
    "number of scales": scales.max()
}
with open(f"{output_path}/{current_timestamp}_log.txt", "w") as f:
    json.dump(log_configs, f, indent=4)



