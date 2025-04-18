import datetime
import os
import pandas as pd


''' ************************************************ CONFIG ************************************************ '''
omni_file_hourly = f'preprocessed_data/omni_data/preprocessed_outliers_removed_omni2_DDJ_kUeKqj.csv'
omni_file_daily = f'preprocessed_data/omni_data/preprocessed_outliers_removed_omni2_daily_1xLBebsoh3.csv'
muon_file_hourly = [f'preprocessed_data/muon_data/preprocessed_1H-intervals_20250227_132422.csv']
muon_file_daily = [f'preprocessed_data/muon_data/preprocessed_1D-intervals_20250227_132422.csv']

current_timestamp = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')

output_path = f'preprocessed_data/prepared_for_correlation'
hourly_output_filename = f'{output_path}/data_hourly_{current_timestamp}.csv'
daily_output_filename = f'{output_path}/data_daily_{current_timestamp}.csv'


''' ********************************************* PROCESSING *********************************************** '''
# Create output path from config if it does not exist.
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Write log.
with open(f'{output_path}/log_{current_timestamp}.txt', 'w') as file:
    file.write(f'Files used:\n'
               f'\tOMNI:\n'
               f'\t\t{omni_file_hourly}\n'
               f'\t\t{omni_file_daily}\n'
               f'\tMuons:\n'
               f'\t\t{muon_file_hourly}\n'
               f'\t\t{muon_file_daily}\n')

# Get hourly muon and omni data.
if omni_file_hourly and muon_file_hourly:
    omni_df_hourly = pd.read_csv(omni_file_hourly, parse_dates=['datetime'])

    # Get hourly muon data.
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

    # Save to CSV.
    hourly_df.to_csv(hourly_output_filename, index=True)
# End if.


# Get daily muon and omni data.
if omni_file_daily and muon_file_daily:
    muon_df_daily = pd.DataFrame()
    for file in muon_file_daily:
        df = pd.read_csv(file, parse_dates=['Time_stamp'])
        muon_df_daily = pd.concat([muon_df_daily, df], ignore_index=True)

    muon_df_cols = muon_df_daily.columns.tolist()
    muon_df_cols[0] = 'datetime'
    muon_df_daily.columns = muon_df_cols

    # Merge daily data.
    omni_df_daily = pd.read_csv(omni_file_daily, parse_dates=['datetime'])
    daily_df = pd.merge(muon_df_daily, omni_df_daily, on='datetime', how='inner')
    daily_df.set_index('datetime', inplace=True)

    # Save to CSV.
    daily_df.to_csv(daily_output_filename, index=True)
# End if.
