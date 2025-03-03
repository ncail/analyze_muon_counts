# Imports.
from datetime import timedelta
import pandas as pd

''' ************************************************ Config ************************************************ '''
data_lst = ('C:\\data\\project_of_excellence\\muon_project_programs\\data_files\\omni_data'
            '/omni2_Iz3WfPPexI.lst')
output_path = f'preprocessed_data/omni_data/preprocessed_hourly_omni2_Iz3WfPPexI.csv'


''' ********************************************* Processing *********************************************** '''
# Get solar data.
data_cols = ['year', 'doy', 'hour', 'Scalar B, nT', 'SW Plasma Speed, km/s', 'R (Sunspot No.)', 'f10.7_index']
df = pd.read_csv(data_lst, delimiter='\s+', header=None, names=data_cols)  # Treat any amount of whitespace as a
# delimiter.

# Convert left 3 columns to one datetime column.
df['datetime'] = pd.to_datetime(df['year'].astype(str) + df['doy'].astype(str).str.zfill(3), format='%Y%j')

# Add hour as a timedelta.
df['datetime'] = df['datetime'] + pd.to_timedelta(df['hour'], unit='h')

# Edit columns.
df.drop(columns=['year', 'doy', 'hour'])
reorder_cols = ['datetime', 'Scalar B, nT', 'SW Plasma Speed, km/s', 'R (Sunspot No.)', 'f10.7_index']
df = df[reorder_cols]

# Save to CSV.
df.to_csv(f'{output_path}', index=False)












