import numpy as np

# --- Configuration ---
np.random.seed(42)  # For reproducibility

# Data structure and simple
city_names = ['Berlin', 'Hamburg', 'Ingolstadt', 'Frankfurt']
day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
time_names = ['morning', 'afternoon', 'evening']

# --- Step 0: Simulate initial dataset (7 days, 4 cities, 3 times per day) ---
weather_data = np.random.randint(5, 25, size=(7, 4, 3))

# --- 2.1: Monday’s morning temperature in Ingolstadt ---
# Indices: Monday=1, Ingolstadt=2, morning=0
ingolstadt_monday_morning = weather_data[1, 2, 0]
print(f"2.1 – Monday's morning temperature in Ingolstadt: {ingolstadt_monday_morning} deg C")

# --- 2.2: Days when Berlin's morning temp < 10°C ---
# Indices: Berlin=0, morning=0
berlin_morning_temps = weather_data[:, 0, 0]
cold_days_berlin = np.argwhere(berlin_morning_temps < 10)
print("2.2 – Days when Berlin had morning temp < 10°C:\n", cold_days_berlin)

# --- 2.3: Add Munich's temperature data as the 5th city ---
munich_data = np.random.randint(5, 25, size=(7, 3))  # shape: (7, 3)
# Reshape to (7, 1, 3) for concatenation along city axis
munich_data_expanded = munich_data[:, np.newaxis, :]
weather_data = np.concatenate((weather_data, munich_data_expanded), axis=1)
city_names.append('Munich')
print("2.3 – New shape after adding Munich:", weather_data.shape)
print(weather_data)


# --- 2.4: Daily average temperature per city (shape: 7, 5) ---
# Average over time axis (axis=2)
daily_city_avg = weather_data.mean(axis=2)
print("2.4 – Daily average temp (7 days × 5 cities):\n", daily_city_avg)

# --- 2.5: City with the lowest weekly average temp ---
weekly_city_avg = daily_city_avg.mean(axis=0)  # shape: (5,)
lowest_temp = np.min(weekly_city_avg)
coldest_cities = [city_names[i] for i, avg in enumerate(weekly_city_avg) if np.isclose(avg, lowest_temp)]
print("2.5.1 – City/cities with the lowest weekly average temperature:", coldest_cities) 