# src/load_shedding_manager.py
import pandas as pd
import numpy as np
import joblib

SEQ_LEN = 24

class LoadSheddingManager:
    def __init__(self, model_path, scaler_path, threshold_watts=5000):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold_watts
        self.appliance_priority = ['Robot_W', 'TV_W', 'Lights_W', 'Fridge_W', 'WaterHeater_W', 'HVAC_W']

    def predict_next_hour(self, recent_history):
        """Predict power for the next hour."""
        recent_history_scaled = self.scaler.transform(recent_history.reshape(-1, 1))
        # Flatten input to match MLP expectations (1, SEQ_LEN)
        X_input = recent_history_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN)
        pred_scaled = self.model.predict(X_input)
        return self.scaler.inverse_transform([[pred_scaled[0]]])[0][0]

    def get_shedding_plan(self, current_loads, predicted_load):
        """Determine which appliances to shed."""
        plan = {}
        if predicted_load <= self.threshold:
            return plan, "No shedding needed."

        excess = predicted_load - self.threshold
        to_shed = []
        for appliance in self.appliance_priority:
            if current_loads.get(appliance, 0) > 0:
                to_shed.append(appliance)
                excess -= current_loads[appliance]
                if excess <= 0:
                    break

        if not to_shed:
            return plan, f"Critical overload! Predicted load {predicted_load:.0f}W > {self.threshold}W. Cannot shed enough."

        for app in to_shed:
            plan[app] = 0
        return plan, f"Shedding plan: Turn OFF {', '.join(to_shed)}."

# --- Simulation ---
if __name__ == '__main__':
    df = pd.read_csv('data/raw/appliance_data.csv', index_col='Datetime', parse_dates=True)
    recent_data = df['Total_Power_W'].values[-SEQ_LEN*2:]

    manager = LoadSheddingManager('models/nn_energy_model.save', 'models/scaler.save', threshold_watts=4500)
    next_load = manager.predict_next_hour(recent_data)
    current = df[['HVAC_W', 'TV_W', 'Robot_W', 'WaterHeater_W', 'Fridge_W', 'Lights_W']].iloc[-1].to_dict()

    print(f"Predicted load for next hour: {next_load:.0f}W")
    plan, msg = manager.get_shedding_plan(current, next_load)
    print(msg)
    if plan:
        print("Suggested actions:", plan)
