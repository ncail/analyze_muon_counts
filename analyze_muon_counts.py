# Imports.
import argparse
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
from scipy.fft import fft
from scipy.signal import welch
import matplotlib.pyplot as plt
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments from user.")
    parser.add_argument('--config', type=str,
                        default='config.json')
    return parser.parse_args()


def load_configs(file_path):
    try:
        with open(file_path, 'r') as file:
            user_config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file '{file_path}' not found.")
    return user_config


# Program start.
def main(args):

    ''' *************************************** INITIALIZE ****************************************** '''
    # Get configs.
    config = load_configs(args.config)

    # Get input and output paths.
    data_filepath = config["data"]["filepath"]

    if config["output"]["path"]:
        output_path = config["output"]["path"]
    else:  # Default path.
        output_path = 'results'

    if config["output"]["filename"]:
        output_filename = config["output"]["filename"]
    else:  # Default filename.
        output_filename = f'results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    # Set time interval to sum event counts over.
    time_interval_config = config['processing']['time_interval']
    time_interval = timedelta(seconds=time_interval_config)

    # Set parameter for Welch's method for getting PSD.
    welch_segment_length = config['processing']['welch_segment_length']

    ''' *************************************** PROCESSING ****************************************** '''
    # Read file into dataframe. File should be muon detector CSV file.
    df = pd.read_csv(data_filepath)

    # Delete invalid rows, convert data to pandas standards:
    # timestamp column to datetime, and count column to numeric.
    column_names = df.columns
    timestamp_col = column_names[0]
    count_col = column_names[1]
    df.drop(index=0, inplace=True)  # First row is a sub-header.
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])  # Convert timestamp col to datetime.
    df[count_col] = pd.to_numeric(df[count_col])  # Convert count col to numeric.

    # Create new dataset start.
    # Create [time interval] counts and store in new dataframe, with start datetime of each total count.
    df.set_index(timestamp_col, inplace=True)  # Required for pd.resample().

    # New dataframe has timestamp col, and count col.
    # Timestamp col stores start time of each interval.
    # Count col stores the last-count-value - first-count-value over the time interval.
    time_avgd_df = df[count_col].resample(time_interval).apply(
        lambda x: x.iloc[-1] - x.iloc[0])
    # Create new dataset end.

    # Convert the event counts to a numpy array.
    event_counts_raw = time_avgd_df.values

    # Remove expected constant signal/background noise.
    background_level = np.mean(event_counts_raw)
    event_counts_detrended = event_counts_raw - background_level

    # Apply Fourier Transform on both arrays.
    fft_values_raw = fft(event_counts_raw)
    fft_values_detrended = fft(event_counts_detrended)

    # Get the frequencies associated with the FFT results.
    # We should expect to see a spike on the 0 frequency for the raw data, representing the mean
    # background radiation, so higher frequency components may be harder to identify in the ft
    # spectrum.
    n = len(event_counts_raw)
    frequencies = np.fft.fftfreq(n, d=int(time_interval.total_seconds()))

    # Get magnitudes of frequency components: shows the strength of each frequency component.
    magnitude_raw = np.abs(fft_values_raw)
    magnitude_detrended = np.abs(fft_values_detrended)

    # Welch's method to get power spectral density (PSD) on a smoothed signal.
    sampling_rate = 1/time_interval.total_seconds()
    frequencies_welch, power_density = welch(event_counts_detrended, fs=sampling_rate, nperseg=welch_segment_length)

    ''' ****************************************** RESULTS ****************************************** '''
    # Send corresponding results to CSV file.
    # fft_df = pd.DataFrame({
    #     'Frequency (Hz)': frequencies,
    #     'Raw Magnitude': magnitude_raw,
    #     'Detrended Magnitude': magnitude_detrended
    # })
    # fft_df.to_csv(f'{output_path}/{output_filename}.csv', index=False)

    # Get timestamps for plotting raw data.
    time_series = time_avgd_df.index.tolist()

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
    plt.ylabel(f"Event count (per {int(time_interval.total_seconds())} sec)")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot raw data end.

    # Plot power spectrum start.
    f_24hr = 1 / (24 * 3600)

    plt.figure(figsize=(12, 6))
    plt.semilogy(frequencies_welch, power_density)
    plt.axvline(x=f_24hr, color='g', linestyle='--', label='24-hour frequency')

    # Draw vertical lines at x-axis indices with frequency value in per hours.
    frequency_grid_lines = np.arange(0, max(frequencies_welch) + max(frequencies_welch)/5, max(frequencies_welch)/5)
    for frequency in frequency_grid_lines:
        if frequency != 0:
            time_in_hours = (1 / frequency) / 3600  # Hours conversion.
            plt.axvline(x=frequency, color='k', linestyle='--')
            ymin, ymax = min(power_density), max(power_density)
            middle_log = (np.log10(ymin) + np.log10(ymax)) / 2
            middle_y = 10 ** middle_log
            plt.text(frequency, middle_y, f'{round(time_in_hours, 2)} hours', ha='right', va='bottom')

    # Draw and label frequency of max power in per hours.
    max_power_pos = power_density.argmax()
    max_power_frequency = float(frequencies_welch[max_power_pos])
    max_power_freq_in_hours = (1 / max_power_frequency) / 3600
    plt.axvline(x=max_power_frequency, color='m', linestyle='--',
                label=f'Maximum power: {round(max_power_freq_in_hours, 2)}-hour frequency')
    # plt.text(max_power_frequency, 1.7e6, f'{round(max_power_freq_in_hours, 2)} hours', ha='right', va='bottom')

    plt.title("Power Spectral Density of Detrended Muon Counts")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid()
    plt.show()
    # Plot power spectrum end.

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


if __name__ == "__main__":
    main_args = parse_arguments()
    main(main_args)









