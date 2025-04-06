"""
Calculates a harmonic fit (amplitude and phase) for 24-hour hourly muon count data and omni data.
Omni data in hourly format are IMF and SWS. Daily averaged Omni data are IMF, SWS, Sunspot No., and F10.7 Index.
Harmonic fit will be done on hourly data only.
The resulting amplitudes and phases can be used to calculate the Pearson and Spearman correlations for muon counts vs
IMF, and muon counts vs SWS.
Daily Omni data will be correlated directly with daily-binned muon counts.
Muon and Omni data are in UTC, so peak hour is calculated in UTC, and then converted to local time dynamically so
that Daylight Savings Time is accounted for.
"""


# Imports.
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import seaborn as sns

''' ************************************************ CONFIG ************************************************ '''
omni_file_hourly = f'preprocessed_data/omni_data/preprocessed_hourly_omni2_DDJ_kUeKqj.csv'
omni_file_daily = f'preprocessed_data/omni_data/preprocessed_omni2_daily_1xLBebsoh3.csv'
muon_file_hourly = [f'preprocessed_data/muon_data/preprocessed_1H-intervals_20250227_132422.csv']
muon_file_daily = [f'preprocessed_data/muon_data/preprocessed_1D-intervals_20250227_132422.csv']

current_timestamp = datetime.datetime.now().strftime('%H%M_%m%d%Y')
output_path = f'results/{current_timestamp}'
visuals_path = f'{output_path}/plots'

save_harmonic_plots = False
do_heatmaps = False
do_visuals = True


''' ********************************************* PROCESSING *********************************************** '''
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(visuals_path):
    os.makedirs(visuals_path)

with open(f'{output_path}/log_{current_timestamp}.txt', 'w') as file:
    file.write(f'Files used:\n'
               f'\tOMNI:\n'
               f'\t\t{omni_file_hourly}\n'
               f'\t\t{omni_file_daily}\n'
               f'\tMuons:\n'
               f'\t\t{muon_file_hourly}\n'
               f'\t\t{muon_file_daily}\n')

# Frequency for 24-hour component.
w = 2 * np.pi / 24  # 24-hour frequency.

# Get hourly data.
omni_df_hourly = pd.read_csv(omni_file_hourly, parse_dates=['datetime'])

# Get all muon data.
muon_df_hourly = pd.DataFrame()
for file in muon_file_hourly:
    df = pd.read_csv(file, parse_dates=['Time_stamp'])
    # Start df at first midnight.
    df.set_index('Time_stamp', inplace=True)
    first_midnight = df.index[df.index.hour == 0][0]
    df = df.loc[first_midnight:]
    df = df.reset_index()
    muon_df_hourly = pd.concat([muon_df_hourly, df], ignore_index=True)

muon_df_cols = muon_df_hourly.columns.tolist()
muon_df_cols[0] = 'datetime'
muon_df_hourly.columns = muon_df_cols

# Merge on datetime. Keep only matching rows from omni_df.
hourly_df = pd.merge(muon_df_hourly, omni_df_hourly, on='datetime', how='inner')
hourly_df.set_index('datetime', inplace=True)

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
''' ********************************************* DATA FITTING LOOP *********************************************** '''
amplitudes_df = pd.DataFrame()
phases_df = pd.DataFrame()

titles = ['Muon count', 'Solar wind speed', 'Interplanetary magnetic field']
y_axes = ['Counts per hour', 'Hourly-averaged SWS (km/s)', 'Hourly-averaged IMF (nT)']

col_num = 0
for col in hourly_df.columns[:3]:
    results = []
    for lLoop in range(0, len(hourly_df), 24):
        if lLoop + 24 > len(hourly_df):
            break

        ''' ******************************************** COMPUTE FIT ********************************************** '''
        segment = hourly_df.iloc[lLoop:lLoop+24]
        t = np.arange(24)
        data = segment[col].values

        # Least squares solving for coefficients. These will allow us to extract the amplitude and phase later.
        X = np.column_stack([
            np.cos(w * t),
            np.sin(w * t),
            np.ones_like(t)  # Offset term
        ])

        # Solve least squares.
        coeffs, _, _, _ = np.linalg.lstsq(X, data, rcond=None)
        a0, b0, A = coeffs  # Extract coefficients.

        # Compute amplitude and phase for the 24-hour component.
        B = np.sqrt(a0**2 + b0**2)
        phi = np.arctan2(-b0, a0)
        if phi < 0:
            phi += 2 * np.pi  # Shift to 0-2pi range since -pi to pi range is returned.

        # Compute the fit.
        y_fit = A + B * np.cos(w * t + phi)

        # Calculate the sum of squared differences (quality of fit).
        ss_res = np.sum((data - y_fit) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)  # R-squared for goodness of fit.

        # Convert phase to local time. Hours are in UTC.
        phi_hours = (phi % (2 * np.pi)) * (24 / (2 * np.pi))
        peak_hour = 24 - phi_hours

        # Save results.
        results.append([hourly_df.index[lLoop], A, B, phi, phi_hours, peak_hour, r2])

        ''' ********************************************* PLOT FIT *********************************************** '''
        if save_harmonic_plots:
            # Plot raw and harmonic fit, and save figure.
            time_range = hourly_df.index[lLoop:lLoop + 24].strftime('%m/%d/%Y %H:00')
            plt.figure(figsize=(10, 6))
            plt.scatter(t, data, label='Data')
            plt.plot(t, y_fit, label='Harmonic fit', color='g')
            plt.title(f"{titles[col_num]} 24-hr harmonic fit")
            plt.ylabel(y_axes[col_num])
            plt.axvline(x=peak_hour, color='m', linestyle='--', label=f'Peak hour (CT): {peak_hour:.1f}')
            plt.xticks(t, time_range, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'{visuals_path}/{col}_harmonic_fit_plot_{lLoop // 24}.png', bbox_inches='tight')
            # plt.show()
            plt.close()
    # End current col.
    col_num += 1

    ''' ********************************************* WRITE RESULTS *********************************************** '''
    # Convert to dataframe and write results to csv.
    df_results = pd.DataFrame(results, columns=["Day", "Offset", "Amplitude", "Phase (rad)", "Phase (hours)",
                                                "Peak hour", "R2"])

    # Convert peak hours in UTC to local time.
    temp = pd.DataFrame()

    # Combine day and peak hour, set as UTC.
    temp['datetime_utc'] = df_results['Day'] + pd.to_timedelta(df_results['Peak hour'], unit='h')
    # temp['datetime_utc'] = temp['datetime_utc'].dt.tz_localize('UTC')

    # Convert to local time (auto DST handling).
    temp['datetime_local'] = temp['datetime_utc'].dt.tz_convert(ZoneInfo('America/Chicago'))

    # Extract just the local hour if needed.
    df_results['Peak hour'] = temp['datetime_local'].dt.hour + temp['datetime_local'].dt.minute / 60

    # Save to csv.
    df_results.to_csv(f'{output_path}/{col}_harmonic_components_per_day.csv', index=False)

    # Add amplitudes and phases for current dataset to dataframe.
    amplitudes_df[col] = df_results['Amplitude']
    phases_df[col] = df_results['Peak hour']
# End all cols.


''' ********************************************* CORRELATION CALCS *********************************************** '''
if do_heatmaps:
    # Correlations.
    # Get muon and omni daily data.
    muon_df_daily = pd.DataFrame()
    for file in muon_file_daily:
        df = pd.read_csv(file, parse_dates=['Time_stamp'])
        muon_df_daily = pd.concat([muon_df_daily, df], ignore_index=True)

    # Merge daily data.
    muon_df_daily.columns = muon_df_cols
    omni_df_daily = pd.read_csv(omni_file_daily, parse_dates=['datetime'])
    daily_df = pd.merge(muon_df_daily, omni_df_daily, on='datetime', how='inner')
    daily_df.set_index('datetime', inplace=True)

    # Do pearson and spearman correlation on daily data.
    all_daily_pearson_corr = daily_df.corr(method='pearson')
    all_daily_spearman_corr = daily_df.corr(method='spearman')
    all_daily_pearson_corr.to_csv(f'{output_path}/raw_daily_pearson_corr_matrix.csv')
    all_daily_spearman_corr.to_csv(f'{output_path}/raw_daily_spearman_corr_matrix.csv')

    # Do pearson and spearman correlation on amplitude and phase data.
    amplitudes_pearson_corr = amplitudes_df.corr(method='pearson')
    amplitudes_spearman_corr = amplitudes_df.corr(method='spearman')
    amplitudes_pearson_corr.to_csv(f'{output_path}/amplitudes_pearson_corr_matrix.csv')
    amplitudes_spearman_corr.to_csv(f'{output_path}/amplitudes_spearman_corr_matrix.csv')

    phases_pearson_corr = phases_df.corr(method='pearson')
    phases_spearman_corr = phases_df.corr(method='spearman')
    phases_pearson_corr.to_csv(f'{output_path}/phases_pearson_corr_matrix.csv')
    phases_spearman_corr.to_csv(f'{output_path}/phases_spearman_corr_matrix.csv')

# Record mean and median amplitude and phase to csv.


''' ************************************************ VISUALIZATION ************************************************* '''


def heat_map(corr_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, linecolor='black')
    plt.title(f'{title}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{visuals_path}/{filename}.png', bbox_inches='tight')
    # plt.show()


if do_heatmaps:
    # Heat map for correlation matrices.
    corrs = [all_daily_pearson_corr, all_daily_spearman_corr,
             amplitudes_pearson_corr, amplitudes_spearman_corr,
             phases_pearson_corr, phases_spearman_corr]
    titles = ['Muon count and solar weather variable Pearson correlation',
              'Muon count and solar weather variable Spearman correlation',
              'Muon count, SWS, IMF harmonic fit amplitudes Pearson correlation',
              'Muon count, SWS, IMF harmonic fit amplitudes Spearman correlation',
              'Muon count, SWS, IMF harmonic fit phases Pearson correlation',
              'Muon count, SWS, IMF harmonic fit phases Spearman correlation']
    filenames = ['heatmap_raw_daily_pearson_corr_matrix', 'heatmap_raw_daily_spearman_corr_matrix',
                 'heatmap_amplitudes_pearson_corr_matrix', 'heatmap_amplitudes_spearman_corr_matrix',
                 'heatmap_phases_pearson_corr_matrix', 'heatmap_phases_spearman_corr_matrix']
    for lLoop in range(len(corrs)):
        heat_map(corrs[lLoop], titles[lLoop], filenames[lLoop])
# End if.

if do_visuals:
    # Build df for Count, SWS, and IMF with the amp and phase assigned to the hourly data.
    # For variable
    for col in hourly_df.columns[:3]:
        variable_hourly = hourly_df[col]

        # Get amplitudes.
        variable_amps = amplitudes_df[col]
        variable_phases = phases_df[col]

        # Extend each amplitude to span 24 points (24-hr day).
        repeated_amps = variable_amps.repeat(24).reset_index(drop=True)
        repeated_phases = variable_phases.repeat(24).reset_index(drop=True)
        variable_hourly = variable_hourly.iloc[:len(repeated_amps)]  # There are less amplitudes per day than hourly
        # data, so cut off straggling hourly data.

        # Plot amplitude/phase vs variable.
        plt.figure(figsize=(10, 6))
        plt.scatter(variable_hourly, repeated_amps)
        plt.ylabel(f'Amplitude')
        plt.xlabel(f'{col}')
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(variable_hourly, repeated_phases)
        plt.ylabel(f'Peak hour (CT)')
        plt.xlabel(f'{col}')
        plt.show()
    # End for.
# End if.
