# from scipy.fft import rfft, irfft

# Get path to all data files.

# Read csvs into dataframes, concat into one dataframe.

# Sort dataframe by datetime.

# Create dataframe with full time series: min to max datetime using the sampling time of the data.

# Left join the muon data to this full time series. Missing data will appear as nans.

# Weighted slow fft, giving zero weight to nan regions.

# Create weight mask.  weights = np.where(np.isnan(df["value"]), 0, 1)  # 0 for nans, 1 for valid data.

# For computing the fft, fill nans with zeros.  filled_data = np.nan_to_num(df['value'])

# Compute weighted fft.  fft_coeffs = rfft(filled_data)  # We could pass (filled_data * weights) here but this is
# redundant since nans, weighted zero, were also set to zero.

# Reconstruct the signal.  reconstructed = irfft(fft_coeffs).real

# Apply the weight mask to convert missing regions back to nans.  reconstructed[weights == 0] = np.nan

# Plot original and reconstructed signal.
# plt.figure(figsize=(12, 5))
# plt.plot(df["datetime"], df["value"], label="Original Signal (with NaNs)", linestyle="dashed")
# plt.plot(df["datetime"], reconstructed, label="Reconstructed Signal (IFFT)", alpha=0.8)
# plt.xlabel("Time")
# plt.ylabel("Signal Value")
# plt.title("FFT-Based Reconstruction of Time Series with Missing Data")
# plt.legend()
# plt.show()














