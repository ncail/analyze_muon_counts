import pandas as pd
import matplotlib.pyplot as plt
import requests
import datetime
import os


''' ************************************************ CONFIG ************************************************ '''
current_timestamp = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
output_path = 'results/corr_atmos'
scatterplot_output_filename = f'{output_path}/muon_vs_temp_and_pressure_{current_timestamp}.png'


''' ********************************************* PROCESSING *********************************************** '''
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load muon data.
muon_df = pd.read_csv("preprocessed_data/prepared_for_correlation/data_hourly_04172025_221740.csv",
                      index_col=0, parse_dates=True)
muon_df = muon_df.rename_axis("datetime").reset_index()

# Define location and time range.
latitude, longitude = 27.7, -97.4
start_date = muon_df["datetime"].min().date().isoformat()
end_date = muon_df["datetime"].max().date().isoformat()

# Fetch weather data from Open-Meteo.
url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temperature_2m,pressure_msl"
    "&timezone=auto"
)

response = requests.get(url)
data = response.json()

weather_df = pd.DataFrame({
    "datetime": pd.to_datetime(data["hourly"]["time"], utc=True),
    "temperature_2m": data["hourly"]["temperature_2m"],
    "pressure_msl": data["hourly"]["pressure_msl"]
})

# Merge on datetime.
merged = pd.merge(muon_df, weather_df, on="datetime", how="inner")

# Plot correlations.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(merged["temperature_2m"], merged["Count"], alpha=0.6)
plt.xlabel("Temperature (°C)", fontsize=15)
plt.ylabel("Muon Count (per hour)", fontsize=15)
plt.title("Muon Count vs Temperature", fontsize=20)

plt.subplot(1, 2, 2)
plt.scatter(merged["pressure_msl"], merged["Count"], alpha=0.6)
plt.xlabel("Pressure (hPa)", fontsize=15)
plt.ylabel("Muon Count (per hour)", fontsize=15)
plt.title("Muon Count vs Pressure", fontsize=20)

plt.tight_layout()

plt.savefig(scatterplot_output_filename, bbox_inches='tight')
plt.show()

plt.close()

