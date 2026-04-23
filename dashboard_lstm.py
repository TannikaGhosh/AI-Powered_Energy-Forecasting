# dashboard_lstm.py
"""
Enhanced Dashboard with Neural Network Forecasting & Load Shedding
Modified perfectly to use the hardware-safe Scikit-Learn MLP instead of TensorFlow.
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import joblib

SEQ_LEN = 24

# ---------- Load Data & Model ----------
df = pd.read_csv('data/raw/appliance_data.csv', index_col='Datetime', parse_dates=True)

# Load MLP model and scaler
model = joblib.load('models/nn_energy_model.save')
scaler = joblib.load('models/scaler.save')

# ---------- Helper Functions ----------
def get_latest_sequence(data, seq_len=SEQ_LEN):
    """Return the last `seq_len` values of Total_Power_W as scaled array."""
    last_values = data['Total_Power_W'].values[-seq_len:]
    scaled = scaler.transform(last_values.reshape(-1, 1))
    return scaled.reshape(1, seq_len)

def predict_next_hour():
    """Predict total power for the next hour using MLP."""
    X_input = get_latest_sequence(df)
    pred_scaled = model.predict(X_input)
    return scaler.inverse_transform([[pred_scaled[0]]])[0][0]

def get_shedding_plan(current_loads, predicted_load, threshold):
    """Return list of appliances to shed and a message."""
    priority = ['Robot_W', 'TV_W', 'Lights_W', 'Fridge_W', 'WaterHeater_W', 'HVAC_W']
    if predicted_load <= threshold:
        return [], f"✅ Predicted load ({predicted_load:.0f}W) within threshold. No shedding."
    
    excess = predicted_load - threshold
    to_shed = []
    for app in priority:
        if current_loads.get(app, 0) > 0:
            to_shed.append(app)
            excess -= current_loads[app]
            if excess <= 0:
                break
    if not to_shed:
        return [], f"⚠️ Critical overload! Cannot shed enough to reach {threshold}W."
    
    msg = f"🔌 Shedding plan: Turn OFF {', '.join(to_shed)}. Estimated load after shedding: {predicted_load - sum(current_loads[app] for app in to_shed):.0f}W"
    return to_shed, msg

def get_actual_vs_predicted_plot():
    """Generate plotly figure of actual vs predicted for last 24h."""
    total = df['Total_Power_W'].values
    predictions = []
    for i in range(len(total) - SEQ_LEN - 24, len(total) - SEQ_LEN):
        window = total[i:i+SEQ_LEN]
        scaled = scaler.transform(window.reshape(-1, 1))
        pred_scaled = model.predict(scaled.reshape(1, SEQ_LEN))
        predictions.append(scaler.inverse_transform([[pred_scaled[0]]])[0][0])
    
    actual_last_24 = total[-24:]
    pred_last_24 = predictions[-24:] if len(predictions) >= 24 else predictions
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-24:], y=actual_last_24, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df.index[-24:], y=pred_last_24, mode='lines', name='Predicted (Neural Net)'))
    fig.update_layout(title='Actual vs Predicted Power (Last 24 Hours)', xaxis_title='Time', yaxis_title='Power (Watts)')
    return fig

# ---------- Initialize Dash App ----------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🏠 AI-Powered Home Energy Management Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("📊 Current Appliance Load (Latest Hour)"),
            html.Div(id='appliance-table', style={'fontSize': 18}),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.H3("🔮 Neural Network Forecast & Load Shedding"),
            html.Div(id='forecast-result', style={'fontSize': 18, 'color': 'blue'}),
            html.Label("Shedding Threshold (Watts):"),
            dcc.Slider(id='threshold-slider', min=100, max=8000, step=100, value=4500,
                       marks={500: '0.5kW', 2000: '2kW', 4000: '4kW', 6000: '6kW', 8000: '8kW'}),
            html.Div(id='shedding-plan', style={'marginTop': 20, 'fontSize': 16, 'color': 'red'}),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]),
    
    html.Hr(),
    html.H3("📈 Model Performance (Last 24h)"),
    dcc.Graph(id='actual-vs-predicted', figure=get_actual_vs_predicted_plot()),
    
    html.Hr(),
    html.Div([
        html.H3("💡 How It Works"),
        html.P("The Neural Network learns patterns from the last 24 hours of total power usage to predict the next hour. If the predicted load exceeds your threshold, the dashboard suggests turning off low-priority appliances (Robot → TV → Lights → Fridge → Water Heater → HVAC)."),
        html.P("📌 This simulates real-world demand response and load shedding used by smart grids and industries.")
    ], style={'backgroundColor': '#f0f0f0', 'padding': 15, 'borderRadius': 10})
])

# ---------- Callbacks ----------
@app.callback(
    [Output('appliance-table', 'children'),
     Output('forecast-result', 'children'),
     Output('shedding-plan', 'children')],
    [Input('threshold-slider', 'value')]
)
def update_dashboard(threshold):
    latest = df.iloc[-1]
    appliance_names = {
        'HVAC_W': 'HVAC', 'TV_W': 'Smart TV', 'Robot_W': 'Cleaning Robot',
        'WaterHeater_W': 'Water Heater', 'Fridge_W': 'Refrigerator', 'Lights_W': 'Lighting'
    }
    pred = predict_next_hour()
    forecast_text = f"🔮 Predicted load for next hour: {pred:.0f} W"
    
    current_loads = {col: latest[col] for col in appliance_names.keys()}
    shed_list, shed_msg = get_shedding_plan(current_loads, pred, threshold)
    
    table_rows = []
    for col, name in appliance_names.items():
        if col in shed_list:
            # Highlight this row visually to indicate it is being powered off
            row_style = {'backgroundColor': '#ffebee', 'color': '#c62828', 'fontWeight': 'bold', 'transition': 'background-color 0.5s ease'}
        else:
            row_style = {'transition': 'background-color 0.5s ease'}
            
        table_rows.append(html.Tr([html.Td(name, style={'padding': '8px'}), html.Td(f"{latest[col]} W", style={'padding': '8px'})], style=row_style))
        
    appliance_table = html.Table([html.Tr([html.Th("Appliance", style={'padding': '8px'}), html.Th("Power (W)", style={'padding': '8px'})], style={'backgroundColor': '#f5f5f5'})] + table_rows,
                                 style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse', 'textAlign': 'left'})
    
    return appliance_table, forecast_text, shed_msg

if __name__ == '__main__':
    # Fix port conflicts by running on port 8051
    app.run(debug=True, port=8051)
