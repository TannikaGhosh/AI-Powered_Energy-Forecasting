#  AI-Powered Energy Management System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.x-green)](https://plotly.com/dash/)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-blue)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 What This Project Does (In Simple Terms)

This project builds **three AI dashboards** that help homes, industries, and grid operators save energy and money:

1. **Home Energy Management** – An LSTM neural network predicts your next hour’s electricity usage. If it forecasts a spike, the dashboard recommends turning off low‑priority appliances (robot → TV → lights → fridge → water heater → HVAC). You can adjust the power threshold and simulate a peak load to see load shedding in action.

2. **Sector Carbon Credit Analysis (₹)** – Compares six sectors (Domestic, Food, Pharma, Car, Government, Private). It simulates their unique energy patterns (e.g., festive season peaks for food, new model launches for cars) and calculates **carbon credits in Indian Rupees** based on CO₂ saved by AI forecasting.

3. **Energy Analytics & 3D Risk Dashboard** – Interactive 3D plots (energy vs temperature vs humidity, hour vs day of week), plus live risk gauges for power cuts, short circuits, and excess electricity.

**How to use it:** Run the dashboards, explore scenarios, and see how AI forecasting reduces energy waste and carbon emissions.

**How to modify it:** Change appliance power values, add new sectors, adjust carbon credit price, or connect real weather APIs. All data generation is modular and well‑commented.

---

##  What This Project Solves

- Predicts short-term electricity demand using an AI model.
- Recommends load shedding for non-critical appliances when demand exceeds thresholds.
- Simulates six Indian sectors and computes carbon credits in ? per ton of CO2 saved.
- Visualizes energy, risk, and carbon patterns through interactive Dash dashboards.

---

##  Key Features

- **Home Forecast + Load Shedding**: LSTM/MLP model predicts next-hour power and suggests appliance shedding.
- **Sector Carbon Credits**: Domestic, Food, Pharma, Car, Government, and Private sectors with seasonal patterns.
- **Indian carbon credit pricing**: ?2000 per ton CO2 saved.
- **Interactive dashboards**: threshold control, sector selector, 3D analytics, date filters, and risk gauges.
- **Modular code**: separate scripts for data generation, forecasting, analysis, and dashboarding.

---

##  Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn, TensorFlow
- Dash, Plotly
- Matplotlib, Seaborn
- Flask, Joblib

---

##  Project Structure

```
AI-Powered Energy Consumption Forecasting System/
+-- data/
�   +-- raw/
�   +-- processed/
+-- models/
+-- outputs/
�   +-- figures/
+-- src/
�   +-- advanced_data_generation.py
�   +-- advanced_preprocess.py
�   +-- advanced_train.py
�   +-- analyze_sustainability.py
�   +-- appliance_simulator.py
�   +-- data_generation.py
�   +-- load_shedding_manager.py
�   +-- lstm_forecaster.py
�   +-- preprocess.py
�   +-- sector_analysis.py
�   +-- sector_dashboard.py
�   +-- train_model.py
+-- dashboard_appliance.py
+-- dashboard_3d.py
+-- sector_dashboard.py
+-- requirements.txt
+-- README.md
+-- .gitignore
```

---

##  Dashboards

### 1. Home Energy Management
- `dashboard_appliance.py`
- Includes neural network forecasting and load shedding recommendations.
- Shows predicted vs actual load and a table of appliance statuses.
- Run with:
  ```bash
  python dashboard_appliance.py
  ```
- Open the browser at `http://127.0.0.1:8052/`

### 2. Sector Carbon Credit Analysis
- `sector_dashboard.py` or integrated into `dashboard_appliance.py` under the Sector Carbon Credits tab.
- Compare six sectors to see carbon credits in ? and seasonal peaks.
- Run with:
  ```bash
  python sector_dashboard.py
  ```
- Open the browser at `http://127.0.0.1:8051/`

### 3. Energy Analytics
- `dashboard_appliance.py` includes the Energy Analytics tab with 3D and risk views.
- Also `dashboard_3d.py` can be used for dedicated analytics visualization.

---

##  Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/AI-Energy-Forecasting.git
   cd "AI-Powered Energy Consumption Forecasting System"
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Generate Data & Models

Run the relevant scripts to generate datasets and model artifacts:

```bash
python src/appliance_simulator.py
python src/lstm_forecaster.py
python src/advanced_data_generation.py
python src/sector_analysis.py
```

The last command generates comparison graphs under `outputs/figures/`.

---

##  Requirements

The main dependencies are listed in `requirements.txt` and include Dash and Plotly for the dashboards.

---

##  GitHub Publication Guide

### 1. Initialize the repository

```bash
git init
git add .
git commit -m "Initial commit: AI energy management dashboards with sector carbon credit analysis"
```

### 2. Create a GitHub repo
- Go to [github.com/new](https://github.com/new)
- Name: `AI-Energy-Forecasting`
- Description: `AI-powered energy forecasting, load shedding, and carbon credit dashboards for Indian sectors`
- Make it public

### 3. Add remote and push

```bash
git remote add origin https://github.com/yourusername/AI-Energy-Forecasting.git
git branch -M main
git push -u origin main
```

### 4. Add screenshots
- Save dashboard images under `images/`
- Commit and push the screenshots

### 5. Verify
- README renders correctly on GitHub
- Dashboards start without errors
- Large data/model files are excluded by `.gitignore`

---

##  Final Notes

- The integrated dashboard is titled **Energy Sector Analysis: Carbon Credits in ?**.
- The home dashboard retains **neural network forecasting** and **load shedding**.
- The sector analysis uses realistic Indian assumptions like festive-season spikes and ?2000/ton CO2 pricing.

---

##  Contact

www.linkedin.com/in/tannika-ghosh-0b1497338| ghoshtannikaofficial2002@gmail.com 
