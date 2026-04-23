# src/lstm_forecaster.py
# (Modified to use Scikit-Learn MLP Neural Network to bypass Windows DLL constraints)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt

# 1. Load and Prepare Data
df = pd.read_csv('data/raw/appliance_data.csv', index_col='Datetime', parse_dates=True)
total_power = df[['Total_Power_W']]

# 2. Scale the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(total_power)

# 3. Create Sequences (use last 24 hours to predict next hour)
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].flatten())  # Flattened for MLP
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 24
X, y = create_sequences(scaled_data, SEQ_LEN)

# 4. Train/Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build MLP Neural Network (Hardware-Safe Alternative to TF LSTM)
model = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)

# 6. Train the Model
print("Training Neural Network Forecaster...")
model.fit(X_train, y_train.ravel())

# 7. Evaluate
y_pred_scaled = model.predict(X_test).reshape(-1, 1)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"\nNeural Network Performance - MAE: {mae:.2f}W, RMSE: {rmse:.2f}W")

# 8. Save Model and Scaler
joblib.dump(model, 'models/nn_energy_model.save')
joblib.dump(scaler, 'models/scaler.save')
print("Model and scaler saved successfully.")

# 9. Plot Results
plt.figure(figsize=(14,5))
plt.plot(y_test_actual[-168:], label='Actual')
plt.plot(y_pred[-168:], label='Predicted')
plt.title('Neural Network: Actual vs Predicted Power Consumption (Last Week)')
plt.xlabel('Hour')
plt.ylabel('Power (Watts)')
plt.legend()
plt.savefig('outputs/figures/nn_actual_vs_predicted.png')
