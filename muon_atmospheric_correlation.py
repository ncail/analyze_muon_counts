import requests
import pandas as pd

# Download temperature and pressure data from open-meteo for specified date range, and lat/long of TAMUCC.
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 29.7604,
    "longitude": -95.3698,
    "start_date": "2023-01-01",
    "end_date": "2023-01-03",
    "hourly": "temperature_2m,pressure_msl",
    "timezone": "auto"
}
response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data['hourly'])
print(df.head())











