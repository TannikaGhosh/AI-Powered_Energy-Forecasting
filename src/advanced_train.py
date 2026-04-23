import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

def train_advanced_model():
    df = pd.read_csv('data/processed/advanced_features.csv', index_col='Datetime', parse_dates=True)
    
    # Feature columns (all except target and non‑numeric)
    exclude = ['Energy_kW', 'LowPollution_Flag', 'CarbonCredit_USD']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y = df['Energy_kW']
    
    # Time‑series split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Advanced Model Performance:")
    print(f"MAE: {mae:.3f} kW")
    print(f"R² : {r2:.3f}")
    
    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    importances.head(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('outputs/figures/feature_importance_advanced.png')
    
    joblib.dump(model, 'models/energy_forecast_advanced.pkl')
    print("Advanced model saved.")
    
    return model, mae, r2

if __name__ == '__main__':
    train_advanced_model()
