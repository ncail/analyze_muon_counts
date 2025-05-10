# Imports.
from datetime import datetime
import json
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq
import pywt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


''' ***************************************** CONFIG ******************************************** '''
# Config mode (of plotting results): 0 - raw data, 1 - fft, 2 - wavelet,
# 3 - rolling rate, 4 - histogram.
mode = 2

# Get input and output paths.
data_filepath = 'preprocessed_data/muon_data/preprocessed_1H-intervals_20250227_132422.csv'
output_path = 'results/spectral_analysis_plots'

# Plot the wavelet transform in units of frequency. Else units of scale.
mode_2_freq = True

# x-axis limits for rolling rate mode (zoom in on a region corresponding to a hotspot on the wavelet transform
# spectrogram).
x_limits = 0  # [datetime(2025, 2, 27), datetime(2025, 3, 22)]

# For rolling rate. Show derivative.
show_deriv = False

# Text for labelling the axes of figures.
data_label = 'Detrended Counts (per hour)'

# Config signal smoothing.
gaussian_smoothing_sigma = 0

# Config fft post-processing.
low_pass_filter = 1/(6*3600)  # Cutoff frequency.

# Wavelet. Bandwidth-Center frequency.
wavelet = 'cmor1.5-1.0'

# Indices of top 5 FFT frequencies to plot on wavelet spectogram. Choose from FFT results.
fft_lines = [2]

# Window (number of points) for mode 3 rolling average.
window = 10


# Returns endpoints on a log progression. For the widths of the wavelet transform.
def get_scales():

    return np.geomspace(2, 72, num=100)


''' *************************************** PROCESSING ****************************************** '''
# Get timestamp for filenames.
current_timestamp = datetime.now().strftime('%m%d%Y_%H%M%S')

# Read file into dataframe. File should be muon detector CSV file.
df = pd.read_csv(data_filepath, parse_dates=[0])
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

# Get indices of top 5 strongest frequency components.
top5_indices = np.argsort(magnitudes_detrended)[-5:][::-1]  # Top 5 in descending order.

# Get corresponding frequencies and magnitudes.
top5_freqs = frequencies_domain[top5_indices]
top5_mags = magnitudes_detrended[top5_indices]
freq_in_hours = 1 / frequencies_per_hour

# Wavelet analysis.
scales = get_scales()
if mode == 2:
    coeffs, frequencies_wt = pywt.cwt(event_counts_detrended, scales, f'{wavelet}',
                                      sampling_period=time_interval.total_seconds())

# Add mode for testing different wavelets from CWT webpage.


''' **************************************** PLOT RESULTS *************************************** '''
# Get timestamps for plotting raw data and wavelet transform.
time_series = df[timestamp_col].tolist()


# Plot raw data (and smoothed data) start.
# To identify transient events, which may correspond to solar flares, etc.
if mode == 0:
    # Plot raw data and mean value (background level).
    sns.set_style("whitegrid")  # Soft grid.
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, event_counts_raw, label="Data")
    plt.plot(time_series, np.full(len(time_series), background_level), color="red", label="Average")
    plt.title("Muon Counts Time Series", fontsize=20)
    #plt.xlabel("Time", fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    plt.ylabel(data_label, fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    # plt.grid(True)

    if gaussian_smoothing_sigma:
        plt.title(f'Gaussian-smoothed: sigma={gaussian_smoothing_sigma}')

    plt.savefig(f'{output_path}/{current_timestamp}_raw_data_plot.png', bbox_inches='tight')

    plt.show()
    plt.close()
# Plot raw data end.


# Plot ffts start.
if mode == 1:
    sns.set_style("whitegrid")  # Soft grid.
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
        legend_labels.append(f'Peak {ind + 1}: {freq_in_hours[top5_indices[ind]]:.2f}-hour period')
    # End for.

    legend = plt.legend(handles=handles, labels=legend_labels, loc='best', fontsize=15)
    legend.get_frame().set_edgecolor('black')

    plt.title("FFT of Muon Count Data", fontsize=25)
    plt.xlabel("Frequency (Hz)", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Magnitude", fontsize=20)
    # plt.grid(True, alpha=0.75)

    # if low_pass_filter:
        # plt.title(f"Low pass filter excludes freq>1/{int(1/low_pass_filter)} Hz", fontsize=15)

    plt.savefig(f'{output_path}/{current_timestamp}_fft_plot.png', bbox_inches='tight')

    plt.show()
    plt.close()
# Plot ffts end.


# Plot wavelet transform start.
if mode == 2:
    morlet_const = np.sqrt(2)

    # Plot cone of influence.
    coi = morlet_const * scales

    # Convert COI values to actual time indices.
    # COI is in terms of the scale, so for each scale, we compute the start and end of the cone.
    coi_start = time_series[0] + pd.to_timedelta(coi, unit='h')  # Start of COI at each scale.
    coi_end = time_series[-1] - pd.to_timedelta(coi, unit='h')  # End of COI at each scale.

    fig, ax = plt.subplots(figsize=(10, 6))
    if mode_2_freq:
        pcm = ax.pcolormesh(time_series, frequencies_wt, np.abs(coeffs))
        # ax.set_yscale("log")
        ax.set_ylabel("Frequency (Hz)", fontsize=15)
    else:
        pcm = ax.pcolormesh(time_series, scales, np.abs(coeffs))
        plt.ylim(top=max(scales))
        ax.set_ylabel("Scales (hours)", fontsize=15)

    ax.set_yscale("log")
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_title("Muon Count Wavelet Transform", fontsize=25)
    fig.colorbar(pcm, ax=ax)

    # Plot the strongest freq/period lines (chosen from FFT). Aggregate all the line labels in the loop,
    # then plot final line with the full label after the loop.
    label = 'From FFT: '
    first = True
    for ind in fft_lines:
        if first:
            label += f'\n{freq_in_hours[top5_indices[ind]]:.2f}-hour period'
            first = False
        else:
            label += f'\n{freq_in_hours[top5_indices[ind]]:.2f}-hour period'
            if mode_2_freq:  # Plot the line in units of frequency.
                plt.axhline(y=top5_freqs[ind], color='m', linestyle='--')
            else:  # Plot line in units of scale.
                plt.axhline(y=freq_in_hours[top5_indices[ind]], color='m', linestyle='--')
    # End for.

    # Plot first fft line with full label.
    if mode_2_freq:  # Plot the line in units of frequency.
        plt.axhline(y=top5_freqs[fft_lines[0]], color='m', linestyle='--', label=label)
    else:  # Plot line in units of scale.
        plt.axhline(y=freq_in_hours[top5_indices[fft_lines[0]]], color='m', linestyle='--', label=label)

    # Plot COI.
    valid = coi_start < coi_end
    if mode_2_freq:
        plt.fill_betweenx(frequencies_wt[valid], time_series[0], coi_start[valid],
                          color='white', alpha=0.3, hatch='//')
        plt.fill_betweenx(frequencies_wt[valid], coi_end[valid], time_series[-1],
                          color='white', alpha=0.3, hatch='//')
    else:
        plt.fill_betweenx(scales[valid], time_series[0], coi_start[valid],
                          color='white', alpha=0.3, hatch='//')
        plt.fill_betweenx(scales[valid], coi_end[valid], time_series[-1],
                          color='white', alpha=0.3, hatch='//')

    plt.legend(fontsize=15)
    plt.savefig(f'{output_path}/{current_timestamp}_wavelet_plot.png', bbox_inches='tight')

    plt.show()
    plt.close()
# Plot wavelet transform end.


# Plot rolling rate.
if mode == 3:
    # Use detrended counts since derivative plotting will be centered on zero.
    df[count_col] = event_counts_detrended

    df['lambda_est'] = df[count_col].rolling(window=window, center=True, min_periods=1).mean()
    df['lambda_derivative'] = df['lambda_est'].diff()

    # twin axes for counts, rate; derivative.
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if show_deriv:
        ax2 = ax1.twinx()
        ax2.plot(df[timestamp_col], df['lambda_derivative'], label='dλ/dt', color='green', linestyle=':', alpha=0.6)
        ax2.set_ylabel('dλ/dt', fontsize=15)

    ax1.scatter(df[timestamp_col], df[count_col], label=data_label, alpha=0.4)
    ax1.plot(df[timestamp_col], df['lambda_est'], label='Expected rate λ(t)', color='red')

    if x_limits:
        plt.xlim(left=x_limits[0], right=x_limits[1])

    plt.title(f'Time-varying Muon Count Rate (window={window})', fontsize=20)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=15)
    ax1.set_ylabel(data_label, fontsize=15)
    fig.legend(loc='upper right', bbox_to_anchor=(0.905, 0.89), fontsize=15)

    plt.savefig(f'{output_path}/{current_timestamp}_rolling_rate_plot_{window}_window.png', bbox_inches='tight')

    plt.show()
    plt.close()
# Plot rolling rate end.

if mode == 4:
    def hist(series, x_label, title, filename, bins=0, isPhase=False, indicate_outliers=True):
        # Calculate average and median.
        mean = np.mean(series)
        median = np.median(series)

        # Plot histogram, and mean and median vertical lines.
        plt.figure(figsize=(10, 6))
        if bins:
            plt.hist(series, bins=bins, alpha=0.7, edgecolor='black')
        else:
            plt.hist(series, bins='auto', alpha=0.7, edgecolor='black')
        plt.axvline(x=mean, color='g', linestyle='--', label=f'Mean = {mean:.2f}')
        plt.axvline(x=median, color='g', linestyle='--', label=f'Median = {median:.2f}')

        # Plot shaded outlier region.
        if indicate_outliers:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            plt.axvline(x=lower_bound, color='r', linestyle=':', label=lower_bound)
            plt.axvline(x=upper_bound, color='r', linestyle=':', label=upper_bound)

        plt.xlabel(x_label, fontsize=15)
        if isPhase:
            plt.xlim(left=0, right=24)
        plt.ylabel('Frequency', fontsize=15)
        plt.title(title, fontsize=20)
        plt.legend(fontsize=15)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        plt.close()
    # End def.

    hist(df['Count'], data_label, 'Distribution of Muon Counts',
         f'{output_path}/{current_timestamp}_muon_count_histogram.png')#, bins=200)
# Plot counts histogram end.

''' ***************************************** LOG CONFIGS *************************************** '''
# Log current configs, write to file.
log_configs = {
    "mode [raw data, fft, wavelet]": mode,
    "data filepath": data_filepath,
    "sampling interval": str(time_interval),
    "gaussian smoothing sigma parameter": gaussian_smoothing_sigma,
    "low pass filter cutoff frequency": f'1/{int(1/low_pass_filter)}' if low_pass_filter else 0,
    "wavelet": f'{wavelet}',
    "scale range": [scales.min(), scales.max()],
    "rolling rate window (number of points)": window
}
with open(f"{output_path}/{current_timestamp}_log.txt", "w") as f:
    json.dump(log_configs, f, indent=4)



