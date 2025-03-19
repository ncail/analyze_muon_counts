# Imports.
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

''' ************************************************ CONFIG ************************************************ '''
omni_file = f'preprocessed_data/omni_data/preprocessed_hourly_omni2_Iz3WfPPexI.csv'
muon_file = f'preprocessed_data/muon_data/preprocessed_1H-intervals_20241004_120111_manually_trimmed.csv'
output_path = f'results/harmonic_fit'


''' ********************************************* PROCESSING *********************************************** '''
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Frequency for 24-hour component.
w = 2 * np.pi / 24  # 24-hour frequency.

# Get data.
omni_df = pd.read_csv(omni_file, parse_dates=['datetime'])
muon_df = pd.read_csv(muon_file, parse_dates=['Time_stamp'])
muon_df_cols = muon_df.columns.tolist()
muon_df_cols[0] = 'datetime_CT'
muon_df.columns = muon_df_cols

# Merge on datetime. Keep only matching rows from omni_df.
df = pd.merge(muon_df, omni_df, on='datetime_CT', how='inner')

# Start df at first midnight.
df.set_index('datetime', inplace=True)
first_midnight = df.index[df.index.hour == 0][0]
df = df.loc[first_midnight:]

def test():
    """Test method."""
    np.random.seed(42)
    timestamps = pd.date_range("2025-03-06", periods=24, freq="H")  # Start at midnight
    t_hours = np.arange(len(timestamps))  # Time in hours

    # Generate a signal with known amplitude and phase
    true_A = 10
    true_B = 5
    true_phi = np.pi * 2 * 4 / 24
    omega = 2 * np.pi / 24  # 24-hour cycle

    signal = true_A + true_B * np.cos(omega * t_hours + true_phi) + np.random.normal(0, 0.5, len(t_hours))

    # Step 2: Set up the least squares fit
    X = np.column_stack((np.ones_like(t_hours), np.cos(omega * t_hours), np.sin(omega * t_hours)))
    params, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)

    A_fit, a_fit, b_fit = params

    # Step 3: Compute amplitude and phase
    B_fit = np.sqrt(a_fit**2 + b_fit**2)
    phi_fit = np.arctan2(-b_fit, a_fit)
    if phi_fit < 0:
        phi_fit += 2 * np.pi

    # Step 4: Convert phase to local time.
    phi_hours = (phi_fit % (2 * np.pi)) * (24 / (2 * np.pi))  # Convert radians to hours.
    peak_hour = 24 - phi_hours

    # Print results
    print(f"Extracted Amplitude: {B_fit:.3f}")
    print(f"Extracted Phase (radians): {phi_fit:.3f}")
    print(f"Extracted Phase (hours): {phi_hours:.3f}")

    # Check if the phase correctly corresponds to the original signal
    print("\nExpected vs Extracted:")
    print(f"True Amplitude: {true_B}, Extracted: {B_fit:.3f}")
    print(f"True Phase (hours): {true_phi * 24 / (2 * np.pi):.3f}, Extracted: {phi_hours:.3f}")
    print(f"Peak hour (local time): {peak_hour}")

    y_fit = B_fit * np.cos(omega * t_hours + phi_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, signal - true_A, label='true signal')
    plt.plot(timestamps, y_fit, label='harmonic fit')
    plt.xticks(timestamps, timestamps.strftime('%H'))
    plt.legend()
    plt.show()

    exit()
# End test().


# For each parameter dataset, except daily averaged values sunspot number and f10.7 index.
# For every 24 data points (24 hours), get amplitudes and phases using single harmonic fit.
for col in df.columns[0:2]:
    results = []
    for lLoop in range(0, len(df), 24):
        if lLoop + 24 > len(df):
            break

        segment = df.iloc[lLoop:lLoop+24]
        t = np.arange(24)
        data = segment[col].values

        # Least squares solving for coefficients. These will allow us to extract the amplitude and phase later.
        X = np.column_stack([
            np.cos(w * t), np.sin(w * t),  # 24-hour component
            np.ones_like(t)  # Offset term
        ])

        # Solve least squares.
        coeffs, _, _, _ = np.linalg.lstsq(X, data, rcond=None)
        a0, b0, A = coeffs  # Extract coefficients.

        # Compute amplitude and phase for the 24-hour component.
        B = np.sqrt(a0 ** 2 + b0 ** 2)
        phi = np.arctan2(-b0, a0)
        if phi < 0:
            phi += 2 * np.pi  # Shift to 0-2pi range since -pi to pi range is returned.

        # Compute the fit.
        y_fit = A + B * np.cos(w * t + phi)

        # Calculate the sum of squared differences (quality of fit).
        ss_res = np.sum((data - y_fit) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)  # R-squared for goodness of fit.

        # Convert phase to local time.
        phi_hours = (phi % (2 * np.pi)) * (24 / (2 * np.pi))
        peak_hour = 24 - phi_hours

        # Save results.
        results.append([lLoop // 24, A, B, phi, phi_hours, peak_hour, r2])

        # plt.figure(figsize=(10, 6))
        # plt.plot(t, data, label='data')
        # plt.plot(t, y_fit, label='harmonic fit')
        # plt.title(f"Day {lLoop // 24}: {col}")
        # plt.legend()
        # plt.show()

    # Convert to dataframe and write results to csv.
    df_results = pd.DataFrame(results, columns=["Day", "Offset", "Amplitude", "Phase (rad)", "Phase (hours)",
                                                "Peak hour (local time)", "R2"])
    df_results.to_csv(f'{output_path}/{col}_harmonic_components_per_day.csv', index=False)


# Plot amplitudes vs time, phases vs time, overlapping all correlation parameters and muon data.


# Calculate Pearson and Spearman coefficients between muon amps/phases and correlation parameters.













