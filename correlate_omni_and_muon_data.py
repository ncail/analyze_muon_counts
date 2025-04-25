"""
Harmonic fitting.
Calculates a harmonic fit (amplitude and phase) for 24-hour hourly muon count data and omni data.
Omni data in hourly format are IMF and SWS. Daily averaged Omni data are IMF, SWS, R Sunspot No., and F10.7 Index.
Harmonic fit will be done on hourly data only.
Muon and Omni data are in UTC, so peak hour (from the harmonic phase component) is calculated in UTC,
and then converted to local time dynamically so that Daylight Savings Time is accounted for.

Correlational analysis.
The resulting amplitudes and phases can be used to calculate the Pearson and Spearman correlations for muon counts vs
IMF and SWS.
Daily Omni data will be correlated directly with daily-binned muon counts to produce heatmap visuals of the
Pearson and Spearman correlation matrices.
Since the literature suggests that CR flux should correlate with IMF most strongly one day later, we shift the IMF
back a day and recompute the correlation coefficients and do visualization on this.

Visualization.
The correlation of harmonic components and daily raw data with the muon count are also demonstrated by scatter plots
and line plots.
Scatter plots have the daily muon amplitude or phase versus the respective harmonic component of IMF and SWS,
and versus the daily-averaged values of each of the solar weather variables.
Line plots show the daily amplitudes or phases of muon count, IMF, and SWS over time. And the daily raw muon count,
IMF, SWS, R SNo., and F10.7 Index over time.
Histograms are generated to display the distribution of raw muon counts, and amplitudes and phases, as well as the
average and median as vertical lines.

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


''' ************************************************ CONFIG ************************************************ '''
hourly_data_file = 'preprocessed_data/prepared_for_correlation/data_hourly_04192025_230731_20250325-20250415_whole_days.csv'
daily_data_file = 'preprocessed_data/prepared_for_correlation/data_daily_04192025_230731_20250325-20250415_whole_days.csv'

current_timestamp = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
output_path = f'results/omni_correlation/{current_timestamp}_03252025-04152025'
visuals_path = f'{output_path}/plots'

save_harmonic_plots = False
do_heatmaps = True
do_scatterplots = True
do_lineplots = True
write_corrs = True
do_one_day_shift = False


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

# amplitudes_df = pd.DataFrame()
# phases_df = pd.DataFrame()

titles = ['Muon Count', 'Interplanetary Magnetic Field', 'Solar Wind Speed', 'R Sunspot Number', 'Solar Index F10.7']
y_axes = ['Counts (per hour)', 'Hourly-averaged IMF (nT)', 'Hourly-averaged SWS (km/s)']

# col_num = 0
# for col_num, col in enumerate(hourly_df.columns[:3]):
results = []
for lLoop in range(0, len(hourly_df), 24):
    if lLoop + 24 > len(hourly_df):
        break

    ''' ******************************************** COMPUTE FIT ********************************************** '''
    segment = hourly_df.iloc[lLoop:lLoop+24]
    t = np.arange(24)
    # data = segment[col].values
    data = segment['Count'].values

    if np.isnan(data).any():
        results.append([hourly_df.index[lLoop], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        continue

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
        # plt.title(f"{titles[col_num]} 24-hr Harmonic Fit")
        # plt.ylabel(y_axes[col_num])
        plt.title(f"Muon Count 24-hr Harmonic Fit")
        plt.ylabel(y_axes[0])
        plt.axvline(x=peak_hour, color='m', linestyle='--', label=f'Peak hour (UTC): {peak_hour:.1f}')
        plt.xticks(t, time_range, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.75)
        plt.tight_layout()
        # plt.savefig(f'{visuals_path}/{col}_harmonic_fit_plot_{lLoop // 24}.png', bbox_inches='tight')
        plt.savefig(f'{visuals_path}/harmonic_fit_plot_{lLoop // 24}.png', bbox_inches='tight')
        # plt.show()
        plt.close()
# End current day.

''' ********************************************* WRITE RESULTS *********************************************** '''
# Convert to dataframe and write results to csv.
harmonic_results_df = pd.DataFrame(results, columns=["Day", "Offset", "Amplitude", "Phase (rad)", "Phase (hours)",
                                            "Peak hour", "R2"])

# Save to csv.
# df_results.to_csv(f'{output_path}/{col}_harmonic_components_per_day.csv', index=False)
daily_corr_df = daily_df.copy()  # Check if datetime is index.
daily_corr_df['Count_amplitude'] = harmonic_results_df['Amplitude']
daily_corr_df['Count_peak_hour'] = harmonic_results_df['Peak hour']
daily_corr_df['Count_r2'] = harmonic_results_df['R2']

# Add sine and cosine of phase radians to df so circular average can be done in Excel.
daily_corr_df['sine'] = np.sin(harmonic_results_df['Phase (rad)'])
daily_corr_df['cosine'] = np.cos(harmonic_results_df['Phase (rad)'])

daily_corr_df.to_csv("daily_data_with_harmonic_results.csv", index=True)

# Drop columns that should not be included in correlation heat map for correlational analysis.
daily_corr_df.drop(columns=['Count_r2', 'Count_peak_hour', 'sine', 'cosine'], inplace=True)

# Add amplitudes and phases for current dataset to dataframe.
# amplitudes_df[col] = df_results['Amplitude']
# phases_df[col] = df_results['Peak hour']

# Save amplitudes and phases to csv.
# amplitudes_df.index = daily_df.index
# phases_df.index = daily_df.index
# amplitudes_df.to_csv(f'{output_path}/amplitudes_per_day.csv', index=True)
# phases_df.to_csv(f'{output_path}/phases_per_day.csv', index=True)

''' ********************************************* CORRELATION CALCS *********************************************** '''
if do_one_day_shift:
    count_col = daily_corr_df['Count'].copy()
    daily_corr_df = daily_corr_df.shift(-1)
    daily_corr_df['Count'] = count_col

    # count_amps = amplitudes_df['Count'].copy()
    # amplitudes_df = amplitudes_df.shift(-1)
    # amplitudes_df['Count'] = count_amps

    # count_phases = phases_df['Count'].copy()
    # phases_df = phases_df.shift(-1)
    # phases_df['Count'] = count_phases
# End if.

# Correlations.
# Do pearson and spearman correlation on daily data.
all_daily_pearson_corr = daily_corr_df.corr(method='pearson')
all_daily_spearman_corr = daily_corr_df.corr(method='spearman')

# Do pearson and spearman correlation on amplitude and phase data.
# amplitudes_pearson_corr = amplitudes_df.corr(method='pearson')
# amplitudes_spearman_corr = amplitudes_df.corr(method='spearman')
# Get amp v variables df.
amp_v_variables_df = daily_corr_df.copy()
# amp_v_variables_df.drop(columns=['Count', 'Count_peak_hour'])
reorder_cols = ['Count_amplitude'] + daily_df.columns[1:]
amp_v_variables_df = amp_v_variables_df[reorder_cols]
amp_v_vars_pearson_corr = amp_v_variables_df.corr(method='pearson')
amp_v_vars_spearman_corr = amp_v_variables_df.corr(method='spearman')

# phases_pearson_corr = phases_df.corr(method='pearson')
# phases_spearman_corr = phases_df.corr(method='spearman')
# Do circular correlation for analyzing phase correlation.
phase_v_variables_df = daily_corr_df.copy()
phase_v_variables_df['phase_rad'] = harmonic_results_df['Phase (rad)']
reorder_cols = ['phase_rad'] + daily_df.columns[1:]
phase_v_variables_df = phase_v_variables_df[reorder_cols]

# Test.
import pycircstat  # Mardia's method for circular-linear correlation.
phase_corr = {}
for col in phase_v_variables_df.columns[1:]:
    phase_corr[col] = pycircstat.corrcl(phase_v_variables_df['phase_rad'], phase_v_variables_df[col])


if write_corrs:
    all_daily_pearson_corr.to_csv(f'{output_path}/daily_pearson_corr_matrix.csv')
    all_daily_spearman_corr.to_csv(f'{output_path}/daily_spearman_corr_matrix.csv')
    # amplitudes_pearson_corr.to_csv(f'{output_path}/amplitudes_pearson_corr_matrix.csv')
    # amplitudes_spearman_corr.to_csv(f'{output_path}/amplitudes_spearman_corr_matrix.csv')
    # phases_pearson_corr.to_csv(f'{output_path}/phases_pearson_corr_matrix.csv')
    # phases_spearman_corr.to_csv(f'{output_path}/phases_spearman_corr_matrix.csv')


''' ************************************************ VISUALIZATION ************************************************* '''
daily_col_names = ['MC', 'IMF', 'SWS', 'R SNo.', 'F10.7']
hourly_col_names = ['MC', 'IMF', 'SWS']


def heat_map(corr_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, linecolor='black', square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'{title}', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{visuals_path}/{filename}.png', bbox_inches='tight')
    # plt.show()
# End def.


def lineplot(df, pearson_corr, spearman_corr, data_labels, title, filename):  #, isPhase=False):
    # Customize style.
    sns.set_style("whitegrid")  # Soft grid.
    colors = [mcolors.CSS4_COLORS['royalblue'], mcolors.CSS4_COLORS['darkorchid'], mcolors.CSS4_COLORS['yellowgreen'],
              mcolors.CSS4_COLORS['darkorange'], mcolors.CSS4_COLORS['darkturquoise']]
    # colors = sns.color_palette('bright', n_colors=5)
    markers = ['o', 's', '^', 'D', 'X']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    legend_entries = []
    plt.figure(figsize=(10, 6))
    for ind, col in enumerate(df.columns):
        # Get normalized data [0,1].
        # if isPhase:
        #     normalized = df[col]
        # else:
        normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Plot and also get legend entries.
        if ind > 0:
            plt.plot(df.index, normalized, color=colors[ind], #alpha=0.75, #marker=markers[ind],
                     linestyle=linestyles[ind])
            legend_entries.append(f"{data_labels[ind]}\n"
                                  f"P = {pearson_corr.loc[0, col]:.2f} "
                                  f"S = {spearman_corr.loc[0, col]:.2f}")
        else:
            plt.plot(df.index, normalized, color=colors[ind], lw=2.5, #marker=markers[ind],
                     linestyle=linestyles[ind])
            legend_entries.append(f'{data_labels[ind]}')
    # End for.
    # plt.show()

    legend_handles = [Line2D([0], [0], color=colors[i]) for i in range(len(colors))]
    plt.legend(legend_handles, legend_entries, loc='center left', bbox_to_anchor=(1, 0.5),
               frameon=True, fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    # if isPhase:
    #     plt.ylabel('Peak Hour (UTC)', fontsize=15)
    #     plt.ylim(bottom=0, top=24)
    plt.title(f"{title}", fontsize=20)
    plt.tight_layout()
    # plt.subplots_adjust(right=0.75)
    plt.savefig(filename, bbox_inches='tight')

    #plt.show()
    plt.close()
# End def.


def scatterplot(x_series, y_series, x_label, y_label, title, filename, isPhase=False):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_series, y_series, marker='D')#, alpha=0.6)
    plt.title(title, fontsize=20)
    plt.ylabel(y_label, fontsize=15)
    if isPhase:
        plt.ylim(bottom=0, top=24)
    plt.xlabel(x_label, fontsize=15)
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close()
# End def.


if do_scatterplots:
    # Plot MC amplitude/phase vs variable.
    x_axes = ['Daily-Averaged IMF (nT)', 'Daily-Averaged SWS (km/s)', 'R Sunspot No.', 'F10.7 (sfu)']

    # Iterate over daily averaged data.
    for ind, col in enumerate(daily_df.columns[1:]):
        # Plot MC amplitude vs variable.
        # scatterplot(daily_df[col], amplitudes_df['Count'], f'{x_axes[ind]}', f'MC Amplitude',
        #             f"Correlation of Muon Count Amplitude and {titles[ind+1]}",
        #             f'{visuals_path}/amplitude_vs_{col}.png')
        scatterplot(daily_df[col], daily_corr_df['Count_amplitude'], f'{x_axes[ind]}', f'MC Amplitude',
                    f"Correlation of Muon Count Amplitude and {titles[ind + 1]}",
                    f'{visuals_path}/amplitude_vs_{col}.png')
        # Plot MC phase vs variable.
        # scatterplot(daily_df[col], phases_df['Count'], f'{x_axes[ind]}', f'MC Peak Hour (UTC)',
        #             f"Correlation of Muon Count Phase and {titles[ind+1]}",
        #             f'{visuals_path}/phase_vs_{col}.png', isPhase=True)
        scatterplot(daily_df[col], daily_corr_df['Count_peak_hour'], f'{x_axes[ind]}', f'MC Peak Hour (UTC)',
                    f"Correlation of Muon Count Phase and {titles[ind + 1]}",
                    f'{visuals_path}/phase_vs_{col}.png', isPhase=True)
    # End for.

    # Scatterplot amps vs var amps, and phase vs var phase.
    # Iterate over harmonic component data.
    # for ind, col in enumerate(hourly_df.columns[1:3]):
    #     # Plot MC amplitude vs variable amplitude.
    #     scatterplot(amplitudes_df[col], amplitudes_df['Count'],
    #                 f'{hourly_col_names[ind+1]} Amplitude', f'MC Amplitude',
    #                 f"Correlation of Muon Count Amplitude and {titles[ind+1]} Amplitude",
    #                 f'{visuals_path}/mc_amp_vs_{col}_amp.png')
    #     # Plot MC phase vs variable phase.
    #     scatterplot(phases_df[col], phases_df['Count'],
    #                 f'{hourly_col_names[ind+1]} Peak Hour (UTC)', f'MC Peak Hour (UTC)',
    #                 f"Correlation of Muon Count Phase and {titles[ind+1]} Phase",
    #                 f'{visuals_path}/mc_phase_vs_{col}_phase.png', isPhase=True)
    # End for.
# End if.


if do_lineplots:
    # Plot all daily data (column for each variable), list corrs as data labels.
    data_labels = ['MC', 'IMF', 'SWS', 'SNo.', 'F10.7']
    amp_data_labels = ['MC Amp', 'IMF', 'SWS', 'SNo.', 'F10.7']
    lineplot(daily_df, all_daily_pearson_corr, all_daily_spearman_corr, data_labels,
             'Daily Muon and Solar Weather Data (Normalized to [0,1])',
             f'{visuals_path}/daily_data_correlations_plot.png')
    lineplot(amp_v_variables_df, amp_v_vars_pearson_corr, amp_v_vars_spearman_corr, amp_data_labels,
             'Muon Count Amplitudes vs Solar Weather (Normalized to [0,1])',
             f'{visuals_path}/daily_amplitude_correlations_plot.png')
    # lineplot(amplitudes_df, amplitudes_pearson_corr, amplitudes_spearman_corr,
    #          'Daily Muon and Solar Weather Amplitudes (Normalized to [0,1])',
    #          f'{visuals_path}/daily_amplitude_correlations_plot.png')
    # lineplot(phases_df, phases_pearson_corr, phases_spearman_corr,
    #          'Daily Muon and Solar Weather Phases',
    #          f'{visuals_path}/daily_phases_correlations_plot.png', isPhase=True)
# End if.


if do_heatmaps:
    # Rename columns of corr matrices.
    all_daily_pearson_corr.columns = daily_col_names
    all_daily_spearman_corr.columns = daily_col_names
    all_daily_pearson_corr.index = daily_col_names
    all_daily_spearman_corr.index = daily_col_names
    # amplitudes_pearson_corr.columns = hourly_col_names
    # amplitudes_spearman_corr.columns = hourly_col_names
    # amplitudes_pearson_corr.index = hourly_col_names
    # amplitudes_spearman_corr.index = hourly_col_names
    # phases_pearson_corr.columns = hourly_col_names
    # phases_spearman_corr.columns = hourly_col_names
    # phases_pearson_corr.index = hourly_col_names
    # phases_spearman_corr.index = hourly_col_names

    # Heat map for correlation matrices.
    corrs = [all_daily_pearson_corr, all_daily_spearman_corr]
# amplitudes_pearson_corr, amplitudes_spearman_corr,
# phases_pearson_corr, phases_spearman_corr
    corr_titles = ['Muon Count and Solar Weather: Pearson Correlation',
                   'Muon Count and Solar Weather: Spearman Correlation']
# 'Harmonic Fit Amplitudes: Pearson Correlation',
# 'Harmonic Fit Amplitudes: Spearman Correlation',
# 'Harmonic Fit Phases: Pearson Correlation',
# 'Harmonic Fit Phases: Spearman Correlation'
    filenames = ['heatmap_daily_pearson_corr_matrix', 'heatmap_daily_spearman_corr_matrix']
# 'heatmap_amplitudes_pearson_corr_matrix', 'heatmap_amplitudes_spearman_corr_matrix',
# 'heatmap_phases_pearson_corr_matrix', 'heatmap_phases_spearman_corr_matrix'
    for lLoop in range(len(corrs)):
        heat_map(corrs[lLoop], corr_titles[lLoop], filenames[lLoop])
# End if.

