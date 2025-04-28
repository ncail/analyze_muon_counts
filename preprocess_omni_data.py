# Imports.
from datetime import timedelta
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

''' ************************************************ Config ************************************************ '''
data_lst = ('C:\\data\\project_of_excellence\\muon_project_programs\\data_files\\omni_data'
            '/omni2_8Yxgda57Vu_20250226-20250415.lst')
output_path = f'preprocessed_data/omni_data/preprocessed_omni2_8Yxgda57Vu_20250226-20250415.csv'
# iqr_multiplier = 3  # For determining outliers.

''' ********************************************* Processing *********************************************** '''
# Get solar data.
data_cols = ['year', 'doy', 'hour', 'Scalar_B_nT', 'SW_Plasma_Speed_kmps', 'R_Sunspot_No', 'f10_7_index']
df = pd.read_csv(data_lst, delimiter='\s+', header=None, names=data_cols)  # Treat any amount of whitespace as a
# delimiter.

# Convert left 3 columns to one datetime column.
df['datetime'] = pd.to_datetime(df['year'].astype(str) + df['doy'].astype(str).str.zfill(3), format='%Y%j')

# Add hour as a timedelta.
df['datetime'] = df['datetime'] + pd.to_timedelta(df['hour'], unit='h')

# Set as UTC.
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

# Convert to local time.
#df['datetime'] = df['datetime'].dt.tz_convert(ZoneInfo('America/Chicago'))

# Edit columns.
df.drop(columns=['year', 'doy', 'hour'])
reorder_cols = ['datetime', 'Scalar_B_nT', 'SW_Plasma_Speed_kmps', 'R_Sunspot_No', 'f10_7_index']
df = df[reorder_cols]

# Remove invalid values.
df.replace([999, 999.9, 9999], np.nan, inplace=True)

# for col in df.select_dtypes(include='number').columns:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - iqr_multiplier * IQR
#     upper_bound = Q3 + iqr_multiplier * IQR
#     mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
#     df.loc[~mask, col] = np.nan
# End for.

# Save to CSV.
df.to_csv(f'{output_path}', index=False)












