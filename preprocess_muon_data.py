# Imports.
from datetime import timedelta
import pandas as pd

''' ************************************************ Config ************************************************ '''
data_filepath = ('C:\\data\\project_of_excellence\\muon_project_programs\\data_files\\data_from_onedrive'
                 '/data_log_20241004_120111_manually_trimmed.csv')
resampling_time_interval = timedelta(minutes=60)
output_path = f'preprocessed_data/preprocessed_1H-intervals_20241004_120111_manually_trimmed.csv'

''' ********************************************* Processing *********************************************** '''
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

# Drop last count.
time_avgd_df.drop(time_avgd_df.index[-1], inplace=True)

# Write to csv.
time_avgd_df.to_csv(f'{output_path}', index=False)











