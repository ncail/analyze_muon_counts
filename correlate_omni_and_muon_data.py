"""
Harmonic fitting.
Calculates a harmonic fit (amplitude and phase) for 24-hour hourly muon count data.
Daily-averaged Omni data are IMF, SWS, R Sunspot No., and F10.7 Index.
Harmonic fit will be done on muon data only.
Muon and Omni data should be in UTC from preprocessing steps, so resulting peak hour (from the harmonic phase
component) is in UTC.

Correlational analysis.
The Pearson and Spearman correlations will be calculated for the raw daily muon count vs Omni, and the daily muon
amplitude vs Omni.
Circular-linear correlation coefficient is calculated for the daily muon peak hour vs Omni.
Since the literature suggests that CR flux should correlate with IMF most strongly one day later, we include a flag
toggling to shift all Omni back one day before computing the correlation coefficients and doing visualizations.

Visualization.
The correlation of muon harmonic components and Omni are demonstrated by scatter plots and line plots.
Scatter plots have the daily muon amplitude vs the daily-averaged Omni data values.
A polar scatter plot can be generated to show the correlation of muon peak hour and Omni variables.
"""


# Imports.
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pycircstat2 import correlation  # Mardia's method for circular-linear correlation.


''' ************************************************ CONFIG ************************************************ '''
hourly_data_file = \
    'preprocessed_data/prepared_for_correlation/data_hourly_04192025_225732_20250227-20250415_whole_days.csv'
daily_data_file = \
    'preprocessed_data/prepared_for_correlation/data_daily_04192025_225732_20250227-20250415_whole_days.csv'

current_timestamp = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
output_path = f'results/omni_correlation/{current_timestamp}'
visuals_path = f'{output_path}/plots'

save_harmonic_plots = False
do_heatmaps = True
do_scatterplots = True
do_lineplots = True
write_corrs = True
do_one_day_shift = True


''' ********************************************* PROCESSING *********************************************** '''
# Create output paths from config if they do not exist.
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(visuals_path):
    os.makedirs(visuals_path)

# Write data files used log.
with open(f'{output_path}/log_{current_timestamp}.txt', 'w') as file:
    file.write(f'Files used:\n'
               f'\tHourly:\n'
               f'\t\t{hourly_data_file}\n'
               f'\tDaily:\n'
               f'\t\t{daily_data_file}\n')

# Get hourly data.
hourly_df = pd.read_csv(hourly_data_file, index_col=0, parse_dates=[0])

# Get daily data.
daily_df = pd.read_csv(daily_data_file, index_col=0, parse_dates=[0])

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
# Frequency for 24-hour component.
w = 2 * np.pi / 24  # 24-hour frequency.

results = []
for lLoop in range(0, len(hourly_df), 24):
    if lLoop + 24 > len(hourly_df):
        break

    ''' ******************************************** COMPUTE FIT ********************************************** '''
    segment = hourly_df.iloc[lLoop:lLoop+24]
    t = np.arange(24)
    # data = segment[col].values
    data = segment['Count'].values

    # Least squares solving for coefficients. These will allow us to extract the amplitude and phase later.
    X = np.column_stack([
        np.cos(w * t),
        np.sin(w * t),
        np.ones_like(t)  # Offset term.
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

    # Convert phase to peak hour (UTC).
    phi_hours = (phi % (2 * np.pi)) * (24 / (2 * np.pi))
    peak_hour = 24 - phi_hours
    peak_rad = peak_hour * 2 * np.pi / 24

    # Save results.
    results.append([hourly_df.index[lLoop], A, B, phi, phi_hours, peak_hour, peak_rad, r2])

    ''' ********************************************* PLOT FIT *********************************************** '''
    if save_harmonic_plots:
        # Plot raw and harmonic fit, and save figure.
        time_range = hourly_df.index[lLoop:lLoop + 24].strftime('%m/%d/%Y %H:00')
        plt.figure(figsize=(10, 6))
        plt.scatter(t, data, label='Data')
        plt.plot(t, y_fit, label='Harmonic fit', color='g')
        plt.title(f"Muon Count 24-hr Harmonic Fit")
        plt.ylabel('Counts (per hour)')
        plt.axvline(x=peak_hour, color='m', linestyle='--', label=f'Peak hour (UTC): {peak_hour:.1f}')
        plt.xticks(t, time_range, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.75)
        plt.tight_layout()
        plt.savefig(f'{visuals_path}/harmonic_fit_plot_{lLoop // 24}.png', bbox_inches='tight')
        # plt.show()
        plt.close()
# End current day.

''' ********************************************* WRITE RESULTS *********************************************** '''
# Convert to dataframe and write results to csv.
harmonic_results_df = pd.DataFrame(results, columns=["Day", "Offset", "Amplitude", "Phase (rad)", "Phase (hours)",
                                                     "Peak hour", "Peak hour (rad)", "R2"])

# Save all results to csv, so averages can be calculated in Excel.
harmonic_results_df.set_index('Day', inplace=True)  # So that the assignments to harmonic_results_df cols succeed.
daily_corr_df = daily_df.copy()  # Make copy of space weather data.

# Handle fragmented date ranges. Complete the date range by reindexing the dataframes.
full_range = pd.date_range(start=daily_corr_df.index.min(), end=daily_corr_df.index.max(), freq='D')
daily_corr_df = daily_corr_df.reindex(full_range)
harmonic_results_df = harmonic_results_df.reindex(full_range)

# Add harmonic results to space weather data so that all can be examined at once in a CSV.
daily_df['Count_amplitude'] = harmonic_results_df['Amplitude']
daily_df['Count_peak_hour'] = harmonic_results_df['Peak hour']
daily_df['Count_peak_hour_rad'] = harmonic_results_df["Peak hour (rad)"]
daily_df['Count_r2'] = harmonic_results_df['R2']

# Add sine and cosine of phase radians to df so circular average can be done in Excel.
daily_df['sine'] = np.sin(harmonic_results_df["Peak hour (rad)"])
daily_df['cosine'] = np.cos(harmonic_results_df["Peak hour (rad)"])

daily_df.to_csv(f"{output_path}/daily_data_with_harmonic_results.csv", index=True)

''' ********************************************* CORRELATION CALCS *********************************************** '''

# Shift daily_corr_df. All references to the daily space weather data should be to dail_corr_df.
if do_one_day_shift:
    count_col = daily_corr_df['Count'].copy()
    daily_corr_df = daily_corr_df.shift(-1)
    daily_corr_df['Count'] = count_col
# End if.

# Correlations.
# Do pearson and spearman correlation on daily data.
all_daily_pearson_corr = daily_corr_df.corr(method='pearson')
all_daily_spearman_corr = daily_corr_df.corr(method='spearman')

# Do pearson and spearman correlation on amplitude.
# Get amp v variables df.
amp_v_variables_df = daily_corr_df.copy()
amp_v_variables_df['Count_amplitude'] = harmonic_results_df['Amplitude']
reorder_cols = ['Count_amplitude'] + list(daily_corr_df.columns[1:])
amp_v_variables_df = amp_v_variables_df[reorder_cols]
amp_v_vars_pearson_corr = amp_v_variables_df.corr(method='pearson')
amp_v_vars_spearman_corr = amp_v_variables_df.corr(method='spearman')

# Do circular correlation for analyzing phase correlation.
phase_v_variables_df = daily_corr_df.copy()
phase_v_variables_df['phase_rad'] = harmonic_results_df['Phase (rad)']
reorder_cols = ['phase_rad'] + list(daily_corr_df.columns[1:])
phase_v_variables_df = phase_v_variables_df[reorder_cols]

# Test.
phase_corr = {}
for col in phase_v_variables_df.columns[1:]:
    # Drop NaN rows in this data column, dropping the same ones from phase column.
    clean_data = phase_v_variables_df[['phase_rad', col]].dropna()
    phase_corr[col] = correlation.circ_corrcl(clean_data['phase_rad'], clean_data[col])
# End for.
phase_corr_series = pd.Series(phase_corr)

if write_corrs:
    all_daily_pearson_corr.to_csv(f'{output_path}/daily_pearson_corr_matrix.csv', index=True)
    all_daily_spearman_corr.to_csv(f'{output_path}/daily_spearman_corr_matrix.csv', index=True)
    amp_v_vars_pearson_corr.to_csv(f'{output_path}/amp_pearson_corr_matrix.csv', index=True)
    amp_v_vars_spearman_corr.to_csv(f'{output_path}/amp_spearman_corr_matrix.csv', index=True)
    phase_corr_series.to_csv(f'{output_path}/phase_circular_corr_matrix.csv', index=True)

''' ************************************************ VISUALIZATION ************************************************* '''


def heat_map(corr_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'{title}', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
# End def.


def lineplot(df, pearson_corr, spearman_corr, data_labels, title, filename):
    # Customize style.
    sns.set_style("whitegrid")  # Soft grid.
    colors = [mcolors.CSS4_COLORS['royalblue'], mcolors.CSS4_COLORS['darkorchid'], mcolors.CSS4_COLORS['yellowgreen'],
              mcolors.CSS4_COLORS['darkorange'], mcolors.CSS4_COLORS['darkturquoise']]
    # colors = sns.color_palette('bright', n_colors=5)
    markers = ['o', 's', '^', 'D', 'X']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    legend_entries = []
    main_col = df.columns[0]
    plt.figure(figsize=(10, 6))
    for ind, col in enumerate(df.columns):
        # Normalize the data from 0,1.
        normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Plot and also get legend entries. If not main col, add label with corr coeffs.
        if ind > 0:
            plt.plot(df.index, normalized, color=colors[ind], linestyle=linestyles[ind])
            legend_entries.append(f"{data_labels[ind]}\n"
                                  f"P = {pearson_corr.loc[main_col, col]:.2f} "
                                  f"S = {spearman_corr.loc[main_col, col]:.2f}")
        else:  # Give larger line width to main col. Write label of main col without any corr coeffs.
            plt.plot(df.index, normalized, color=colors[ind], lw=2.5, linestyle=linestyles[ind])
            legend_entries.append(f'{data_labels[ind]}')
    # End for.
    # plt.show()

    # Format legend.
    legend_handles = [Line2D([0], [0], color=colors[i]) for i in range(len(colors))]
    plt.legend(legend_handles, legend_entries, loc='center left', bbox_to_anchor=(1, 0.5),
               frameon=True, fontsize=15)

    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"{title}", fontsize=20)
    plt.tight_layout()

    # Save.
    plt.savefig(filename, bbox_inches='tight')

    # plt.show()
    plt.close()
# End def.


def scatterplot(x_series, y_series, x_label, y_label, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_series, y_series, marker='D')#, alpha=0.6)
    plt.title(title, fontsize=20)
    plt.ylabel(y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close()
# End def.


def polar_plot_single(circ_var_radians, linear_var, circ_label, linear_label, title, filename):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot the data.
    ax.scatter(circ_var_radians, linear_var, marker='o', color='b')

    # Set the labels for each hour on the circle.
    ax.set_xticks(np.arange(0, 2*np.pi, np.pi/12))
    ax.set_xticklabels(np.arange(0, 24), fontsize=15)

    ax.set_title(title, va='bottom', fontsize=20)
    ax.set_xlabel(circ_label, fontsize=15)
    ax.set_ylabel(linear_label, fontsize=15, labelpad=30)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close()
# End def.


def polar_plot_multiple(circ_var_radians, linear_vars, circ_label, linear_labels, title, filename):
    colors = [mcolors.CSS4_COLORS['darkorchid'], mcolors.CSS4_COLORS['yellowgreen'],
              mcolors.CSS4_COLORS['darkorange'], mcolors.CSS4_COLORS['darkturquoise']]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot the data.
    for ind, var in enumerate(linear_vars):
        # Normalize from 0,1.
        normalized = (linear_vars[var] - linear_vars[var].min()) / (linear_vars[var].max() - linear_vars[var].min())
        ax.scatter(circ_var_radians, normalized, marker='o', color=colors[ind], label=linear_labels[ind])

    # Set the labels for each hour on the circle.
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 12))
    ax.set_xticklabels(np.arange(0, 24), fontsize=15)

    ax.set_title(title, va='bottom', fontsize=20)
    ax.set_xlabel(circ_label, fontsize=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05), fontsize=15)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
# End def.


if do_scatterplots:
    # Plot MC amplitude vs variable.
    titles = ['Muon Count', 'Interplanetary Magnetic Field', 'Solar Wind Speed', 'R Sunspot Number',
              'Solar Index F10.7']
    x_axes = ['IMF (nT)', 'SWS (km/s)', 'R Sunspot No.', 'F10.7 (sfu)']

    # Iterate over daily averaged data.
    for ind, col in enumerate(daily_corr_df.columns[1:]):  # Excludes 'Count' col. Do not correlate count amplitude
        # to the count itself.
        scatterplot(amp_v_variables_df[col], amp_v_variables_df['Count_amplitude'], f'{x_axes[ind]}', f'MC Amplitude',
                    f"Correlation of Muon Count Amplitude and {titles[ind + 1]}",
                    f'{visuals_path}/amplitude_vs_{col}.png')

        # Do polar scatterplot for phase per variable.
        polar_plot_single(harmonic_results_df['Peak hour (rad)'], daily_corr_df[col],
                   'Peak Hour (UTC)', f'{x_axes[ind]}', f'Peak Hour vs {titles[ind+1]}',
                   f'{visuals_path}/polar_plot_{col}.png')
    # End for.
    # Do polar scatterplot for phase overlapping all variables.
    polar_plot_multiple(harmonic_results_df['Peak hour (rad)'], daily_corr_df.iloc[:, 1:],  # Slice of df: all rows, and cols 1:
               'Peak Hour (UTC)', x_axes, f'Peak Hour vs Space Weather (Normalized to [0,1])',
               f'{visuals_path}/polar_plot_all_variables.png')
# End if.


if do_lineplots:
    # Plot all daily data (column for each variable), list corrs as data labels.
    data_labels = ['MC', 'IMF', 'SWS', 'SNo.', 'F10.7']
    amp_data_labels = ['MC Amp', 'IMF', 'SWS', 'SNo.', 'F10.7']
    lineplot(daily_corr_df, all_daily_pearson_corr, all_daily_spearman_corr, data_labels,
             'Daily Muon and Space Weather Data (Normalized to [0,1])',
             f'{visuals_path}/daily_data_correlations_plot.png')
    lineplot(amp_v_variables_df, amp_v_vars_pearson_corr, amp_v_vars_spearman_corr, amp_data_labels,
             'Muon Count Amplitudes vs Space Weather (Normalized to [0,1])',
             f'{visuals_path}/daily_amplitude_correlations_plot.png')
# End if.


if do_heatmaps:
    # Rename columns of corr matrices.
    daily_col_names = ['MC', 'IMF', 'SWS', 'R SNo.', 'F10.7']
    amp_col_names = ['MC Amp', 'IMF', 'SWS', 'R SNo.', 'F10.7']
    all_daily_pearson_corr.columns,     all_daily_spearman_corr.columns =   daily_col_names, daily_col_names
    all_daily_pearson_corr.index,       all_daily_spearman_corr.index   =   daily_col_names, daily_col_names
    amp_v_vars_pearson_corr.columns,    amp_v_vars_pearson_corr.index   =   amp_col_names, amp_col_names
    amp_v_vars_spearman_corr.columns,   amp_v_vars_spearman_corr.index  =   amp_col_names, amp_col_names

    # Heat map arguments for correlation matrices.
    corrs = [all_daily_pearson_corr, all_daily_spearman_corr,
             amp_v_vars_pearson_corr, amp_v_vars_spearman_corr]
    corr_titles = ['Muon Count vs Space Weather: Pearson Correlation',
                   'Muon Count vs Space Weather: Spearman Correlation',
                   'Muon Count Amplitude vs Space Weather: Pearson Correlation',
                   'Muon Count Amplitude vs Space Weather: Spearman Correlation']
    filenames = [f'{visuals_path}/heatmap_daily_pearson_corr_matrix.png',
                 f'{visuals_path}/heatmap_daily_spearman_corr_matrix.png',
                 f'{visuals_path}/heatmap_amp_pearson_corr_matrix.png',
                 f'{visuals_path}/heatmap_amp_spearman_corr_matrix.png']

    # Make heatmaps.
    for lLoop in range(len(corrs)):
        heat_map(corrs[lLoop], corr_titles[lLoop], filenames[lLoop])
# End if.

