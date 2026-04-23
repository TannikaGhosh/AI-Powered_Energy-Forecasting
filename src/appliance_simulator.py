# src/appliance_simulator.py
import pandas as pd
import numpy as np
from datetime import datetime

def simulate_appliances(start_date='2024-01-01', days=90):
    """Simulate power data for 6 common appliances and calculate total load."""
    date_range = pd.date_range(start=start_date, periods=days*24, freq='h')
    records = []
    np.random.seed(42)

    for dt in date_range:
        hour = dt.hour
        is_weekend = dt.dayofweek >= 5

        # --- 1. HVAC (Heavy load, based on time and temp) ---
        temp_profile = 15 + 10 * np.sin(2 * np.pi * (dt.timetuple().tm_yday - 80) / 365) + 8 * np.sin(2 * np.pi * (hour - 14) / 24)
        hvac_power = 3500 if (temp_profile > 22 and hour > 8 and hour < 20) or (temp_profile < 18 and hour > 17) else 0
        hvac_power = hvac_power * np.random.uniform(0.9, 1.1) # Add some noise

        # --- 2. Smart TV (Evening peak, lower on weekends) ---
        tv_power = 150 if (18 <= hour <= 23) else 0
        if is_weekend: tv_power = tv_power * 0.7
        tv_power = tv_power * np.random.uniform(0.95, 1.05)

        # --- 3. Cleaning Robot (1-2 hours, can be scheduled) ---
        robot_power = 50 if (hour in [10, 14] and not is_weekend) else 0
        robot_power = robot_power * np.random.uniform(0.9, 1.1)

        # --- 4. Water Heater (Morning and evening peaks) ---
        wh_power = 3000 if (hour in [7, 8, 19, 20]) else 0
        wh_power = wh_power * np.random.uniform(0.9, 1.1)

        # --- 5. Refrigerator (Cycles: 40 mins on, 20 mins off) ---
        cycle = (dt.minute // 20) % 3
        fridge_power = 150 if cycle == 0 else 0

        # --- 6. Lighting (On when dark) ---
        light_power = 200 if (hour < 6 or hour > 19) else 0
        if is_weekend: light_power = light_power * 1.2 # More lights on weekends
        light_power = light_power * np.random.uniform(0.9, 1.1)

        # --- Total Power Load (in Watts) ---
        total_power = hvac_power + tv_power + robot_power + wh_power + fridge_power + light_power

        records.append({
            'Datetime': dt,
            'HVAC_W': round(hvac_power),
            'TV_W': round(tv_power),
            'Robot_W': round(robot_power),
            'WaterHeater_W': round(wh_power),
            'Fridge_W': round(fridge_power),
            'Lights_W': round(light_power),
            'Total_Power_W': round(total_power)
        })

    df = pd.DataFrame(records)
    df.set_index('Datetime', inplace=True)
    return df

if __name__ == '__main__':
    df = simulate_appliances()
    df.to_csv('data/raw/appliance_data.csv')
    print("Appliance-level simulation saved to data/raw/appliance_data.csv")
    print(df.head())
