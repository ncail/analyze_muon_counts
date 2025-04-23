"""
Run to merge muon count data with OMNI data after preprocessing both.
Inputs data files corresponding to hourly muon count data, and daily muon count and OMNI data.
Does not convert timestamps of any data to a different timezone.
The preprocessing programs convert timestamps to UTC.
For hourly data: only muon data is processed.
If multiple muon data files are entered, they are concatenated.
cutoff_incomplete flag can be toggled to control if all hours in the hourly data is kept, or if only whole days are
kept.
Whole days are needed for the harmonic analysis in correlate_omni_and_muon_data.py, but all hourly data can be used
for muon_atmospheric_correlation.py.
For daily data: does an inner merge, matching OMNI data entries to the muon count data.
Daily data already only contains complete days since the preprocessing will bin muon count by the day,
and OMNI database outputs one daily-averaged value per day.
"""

import datetime
import os
import pandas as pd
from zoneinfo import ZoneInfo

''' ************************************************ CONFIG ************************************************ '''
# omni_file_hourly = f'preprocessed_data/omni_data/preprocessed_omni2_8Yxgda57Vu_20250226-20250415.csv'
omni_file_daily = f'preprocessed_data/omni_data/preprocessed_omni2_daily_gtNELIevQT_20250226-20250415.csv'
muon_file_hourly = []#f'preprocessed_data/muon_data/preprocessed_1H-intervals_20250227_132422.csv',
                    #f'preprocessed_data/muon_data/preprocessed_1H-intervals_20250325_115858.csv']
muon_file_daily = [f'preprocessed_data/muon_data/preprocessed_1D-intervals_20250227_132422.csv']#,
                   #f'preprocessed_data/muon_data/preprocessed_1D-intervals_20250325_115858.csv']

# Choose to only include complete days of data.
cutoff_incomplete = True

current_timestamp = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')

output_path = f'preprocessed_data/prepared_for_correlation'
hourly_output_filename = f'{output_path}/muon_count_hourly_{current_timestamp}_20250227-20250320_whole_days.csv'
daily_output_filename = f'{output_path}/data_daily_{current_timestamp}_20250227-20250320_whole_days.csv'


''' ********************************************* PROCESSING *********************************************** '''
# Create output path from config if it does not exist.
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Write log.
with open(f'{output_path}/log_{current_timestamp}.txt', 'w') as file:
    file.write(f'Files used:\n'
               f'\tOMNI:\n'
               # f'\t\t{omni_file_hourly}\n'
               f'\t\t{omni_file_daily}\n'
               f'\tMuons:\n'
               f'\t\t{muon_file_hourly}\n'
               f'\t\t{muon_file_daily}\n')

# Get hourly muon and omni data.
# if omni_file_hourly and muon_file_hourly:
if muon_file_hourly:
    # omni_df_hourly = pd.read_csv(omni_file_hourly, parse_dates=['datetime'])
    # omni_df_hourly['datetime'] = omni_df_hourly['datetime'].dt.tz_convert(ZoneInfo('America/Chicago'))

    # Get hourly muon data.
    muon_df_hourly = pd.DataFrame()
    for file in muon_file_hourly:
        df = pd.read_csv(file, parse_dates=['Time_stamp'])
        # df['Time_stamp'] = df['Time_stamp'].dt.tz_convert(ZoneInfo('America/Chicago'))
        # Remove incomplete days of data.
        if cutoff_incomplete:
            # Start df at first midnight.
            df.set_index('Time_stamp', inplace=True)
            first_midnight = df.index[df.index.hour == 0][0]
            df = df.loc[first_midnight:]

            # Cut off last incomplete day.
            # Number of complete days.
            n_complete_days = len(df) // 24
            # Keep only complete days.
            df = df.iloc[:n_complete_days * 24]

            df = df.reset_index()
        # End if.

        muon_df_hourly = pd.concat([muon_df_hourly, df], ignore_index=True)
    # End for.

    # muon_df_cols = muon_df_hourly.columns.tolist()
    # muon_df_cols[0] = 'datetime'
    # muon_df_hourly.columns = muon_df_cols

    # Merge on datetime. Keep only matching rows from omni_df.
    # hourly_df = pd.merge(muon_df_hourly, omni_df_hourly, on='datetime', how='inner')
    # hourly_df.set_index('datetime', inplace=True)
    muon_df_hourly.set_index('datetime', inplace=True)

    # Save to CSV.
    # hourly_df.to_csv(hourly_output_filename, index=True)
    muon_df_hourly.to_csv(hourly_output_filename, index=True)
# End if.


# Get daily muon and omni data.
if omni_file_daily and muon_file_daily:
    muon_df_daily = pd.DataFrame()
    for file in muon_file_daily:
        df = pd.read_csv(file, parse_dates=['Time_stamp'])
        # df['Time_stamp'] = df['Time_stamp'].dt.tz_convert(ZoneInfo('America/Chicago'))
        muon_df_daily = pd.concat([muon_df_daily, df], ignore_index=True)

    muon_df_cols = muon_df_daily.columns.tolist()
    muon_df_cols[0] = 'datetime'
    muon_df_daily.columns = muon_df_cols

    # Merge daily data.
    omni_df_daily = pd.read_csv(omni_file_daily, parse_dates=['datetime'])
    # omni_df_daily['datetime'] = omni_df_daily['datetime'].dt.tz_convert(ZoneInfo('America/Chicago'))
    daily_df = pd.merge(muon_df_daily, omni_df_daily, on='datetime', how='inner')
    daily_df.set_index('datetime', inplace=True)

    # Save to CSV.
    daily_df.to_csv(daily_output_filename, index=True)
# End if.
