# Imports.
import pandas as pd

''' ************************************************ CONFIG ************************************************ '''
omni_file = f'preprocessed_data/omni_data/preprocessed_hourly_omni2_Iz3WfPPexI.csv'
muon_file = f'preprocessed_data/muon_data/preprocessed_1H-intervals_20241004_120111_manually_trimmed.csv'


''' ********************************************* PROCESSING *********************************************** '''
# Get data.
omni_df = pd.read_csv(omni_file, parse_dates=['datetime'])
muon_df = pd.read_csv(muon_file, parse_dates=['Time_stamp'])
muon_df_cols = muon_df.columns.tolist()
muon_df_cols[0] = 'datetime'
muon_df.columns = muon_df_cols

# Merge on datetime. Keep only matching rows from omni_df.
df = pd.merge(muon_df, omni_df, on='datetime', how='inner')

# For every 24 data points (24 hours), get amplitudes and phases using double harmonic fit.


    # Calculate sum of squared differences for each day of fitting.


# Plot amplitudes vs time, phases vs time, overlapping all correlation parameters and muon data.


# Calculate Pearson and Spearman coefficients between muon amps/phases and correlation parameters.













