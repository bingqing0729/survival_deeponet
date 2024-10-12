import pandas as pd
import numpy as np

# Load the data
file_path = 'C:\\Users\\bingq\\OneDrive\\Desktop\\bike_sharing\\bike_sharing_hourly.csv'
data = pd.read_csv(file_path)

# Combine 'dteday' and 'hr' to create a full datetime column
data['datetime'] = pd.to_datetime(data['dteday']) + pd.to_timedelta(data['hr'], unit='h')

# Ensure the data is sorted by datetime if not already
data = data.sort_values(by='datetime').reset_index(drop=True)

# Identify missing hours
all_hours = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='H')
data_full = data.set_index('datetime').reindex(all_hours).reset_index()
data_full.rename(columns={'index': 'datetime'}, inplace=True)


data_full['dteday'] = data_full['datetime'].dt.date
data_full['hr'] = data_full['datetime'].dt.hour
data_full['workingday'] = data_full['workingday'].fillna(method='ffill')
data_full['cnt'] = data_full['cnt'].fillna(0)

# First Pass: Impute values for missing entries
for i in range(len(data_full)):
    if pd.isna(data_full.loc[i, 'temp']):
        if i == 0:  # Missing at the beginning
            data_full.loc[i, 'temp'] = data_full.loc[i + 1, 'temp']
        elif i == len(data_full) - 1:  # Missing at the end
            data_full.loc[i, 'temp'] = data_full.loc[i - 1, 'temp']

# Linear interpolation for consecutive missing values
data_full['temp'] = data_full['temp'].interpolate(method='linear')

# Save the imputed data to a new CSV file
output_file_path = 'C:\\Users\\bingq\\OneDrive\\Desktop\\bike_sharing\\bike_sharing_hourly_imputed.csv'
data_full.to_csv(output_file_path, index=False)

# Verify if there are any missing values in 'temp'
missing_temps = data_full['temp'].isna().sum()
print(f"Number of missing temperature values after imputation: {missing_temps}")