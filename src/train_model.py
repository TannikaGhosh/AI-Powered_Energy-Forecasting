"""
Trains Multi-Layer Perceptron Regressor.
Features: hour, day_of_week, is_weekend, lag_1, rolling_mean_3
Target: Energy_kW
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_energy_model(csv_path='../data/processed/energy_features.csv'):
    df = pd.read_csv(csv_path, index_col='Datetime', parse_dates=True)
    
    # Features and target
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'lag_1', 'rolling_mean_3']
    X = df[feature_cols]
    y = df['Energy_kW']
    
    # Split (time‑series aware: first 80% train, last 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Model
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("📊 Model Performance on Test Set:")
    print(f"MAE  : {mae:.3f} kW")
    print(f"RMSE : {rmse:.3f} kW")
    print(f"R²   : {r2:.3f}")
    
    # Save metrics
    os.makedirs('../outputs', exist_ok=True)
    with open('../outputs/metrics.txt', 'w') as f:
        f.write(f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}")
    
    # Plot: Actual vs Predicted
    plt.figure(figsize=(12,5))
    plt.plot(y_test.values[-168:], label='Actual', alpha=0.7)   # last week
    plt.plot(y_pred[-168:], label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption (Last Week)')
    plt.xlabel('Hour')
    plt.ylabel('Energy (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../outputs/figures/actual_vs_predicted.png', dpi=150)
    plt.close()
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10,4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Prediction Error (kW)')
    plt.savefig('../outputs/figures/residuals.png', dpi=150)
    plt.close()
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/energy_forecast_model.pkl')
    print("✅ Model saved to models/energy_forecast_model.pkl")
    
    return model, mae, rmse, r2

if __name__ == '__main__':
    train_energy_model()
