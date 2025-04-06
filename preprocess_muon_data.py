# Imports.
from datetime import timedelta
import pandas as pd
from zoneinfo import ZoneInfo

''' ************************************************ CONFIG ************************************************ '''
data_filepath = ('C:\\data\\project_of_excellence\\muon_project_programs\\data_files\\data_from_onedrive'
                 '/data_log_20250227_132422.csv')
resampling_time_interval = timedelta(hours=1)
output_path = f'preprocessed_data/muon_data/preprocessed_1H-intervals_20250227_132422.csv'


''' ********************************************* PROCESSING *********************************************** '''
# Read file into dataframe. File should be muon detector CSV file.
df = pd.read_csv(data_filepath)  #, parse_dates=['Time_stamp'])

# Delete invalid rows, convert data to pandas standards:
# timestamp column to datetime, and count column to numeric.
column_names = df.columns
timestamp_col = column_names[0]
count_col = column_names[1]

df.drop(index=0, inplace=True)  # First row is a sub-header.

# Timestamps are in local time.
df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='mixed').dt.tz_localize(ZoneInfo('America/Chicago'))

# Convert time to UTC.
df[timestamp_col] = df[timestamp_col].dt.tz_convert('UTC')
df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

df[count_col] = pd.to_numeric(df[count_col])  # Convert count col to numeric.

# Create new dataset start.
df.set_index(timestamp_col, inplace=True)  # Required for pd.resample().

# New dataframe has timestamp col, and count col.
# Timestamp col stores start time of each interval.
# Count col stores the last-count-value - first-count-value over the time interval (previously converted from
# int to seconds).
time_avgd_series = df[count_col].resample(resampling_time_interval).apply(
    lambda x: x.iloc[-1] - x.iloc[0])
# Create new dataset end.

# Has Time_stamp and Count columns
time_avgd_df = time_avgd_series.reset_index().rename(columns={count_col: 'Count'})

# Drop first and last count.
time_avgd_df.drop(time_avgd_df.index[0], inplace=True)
time_avgd_df.drop(time_avgd_df.index[-1], inplace=True)

# Write to csv.
time_avgd_df.to_csv(f'{output_path}', index=False)











