import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------
# LOAD DATA
# ----------------------------
process_df = pd.read_excel("_h_batch_process_data.xlsx")
production_df = pd.read_excel("_h_batch_production_data.xlsx")

process_df["Batch_ID"] = process_df["Batch_ID"].astype(str)
production_df["Batch_ID"] = production_df["Batch_ID"].astype(str)

df = pd.merge(process_df, production_df, on="Batch_ID")

EMISSION_FACTOR = 0.82
df["Carbon_Emission"] = df["Power_Consumption_kW"] * EMISSION_FACTOR

features = df[[
    "Time_Minutes",
    "Temperature_C",
    "Pressure_Bar",
    "Humidity_Percent",
    "Motor_Speed_RPM",
    "Compression_Force_kN",
    "Flow_Rate_LPM",
    "Power_Consumption_kW",
    "Vibration_mm_s"
]]

targets = df[[
    "Hardness",
    "Dissolution_Rate",
    "Content_Uniformity",
    "Carbon_Emission"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, targets, test_size=0.2, random_state=42
)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200))
model.fit(X_train, y_train)

# ----------------------------
# OPTIMIZATION FUNCTION
# ----------------------------
def calculate_score(prediction):
    hardness, dissolution, uniformity, carbon = prediction
    
    score = (
        hardness * 0.3 +
        dissolution * 0.3 +
        uniformity * 0.2 -
        carbon * 0.2
    )
    
    return score


def optimize_parameters(model, sample_input):
    best_score = -999
    best_config = None

    for _ in range(100):
        variation = sample_input + np.random.normal(0, 0.05, sample_input.shape)
        prediction = model.predict(variation)[0]
        score = calculate_score(prediction)

        if score > best_score:
            best_score = score
            best_config = variation

    return best_config, best_score


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🏭 AI Batch Optimization System")
st.subheader("Smart Manufacturing Intelligence Dashboard")

st.sidebar.header("Enter Process Parameters")

time = st.sidebar.slider("Time (Minutes)", 0.0, 200.0, 50.0)
temp = st.sidebar.slider("Temperature (C)", 0.0, 100.0, 30.0)
pressure = st.sidebar.slider("Pressure (Bar)", 0.0, 10.0, 1.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 40.0)
speed = st.sidebar.slider("Motor Speed (RPM)", 0.0, 500.0, 100.0)
force = st.sidebar.slider("Compression Force (kN)", 0.0, 20.0, 3.0)
flow = st.sidebar.slider("Flow Rate (LPM)", 0.0, 10.0, 1.5)
power = st.sidebar.slider("Power Consumption (kW)", 0.0, 10.0, 2.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 20.0, 3.0)

if st.button("Predict & Optimize"):

    user_data = np.array([[ 
        time, temp, pressure, humidity,
        speed, force, flow, power, vibration
    ]])

    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]

    hardness, dissolution, uniformity, carbon = prediction

    st.success("🔍 Prediction Results")

    st.write("Hardness:", round(hardness, 2))
    st.write("Dissolution Rate:", round(dissolution, 2))
    st.write("Content Uniformity:", round(uniformity, 2))
    st.write("Carbon Emission:", round(carbon, 2))

    best_config, best_score = optimize_parameters(model, user_scaled)
    optimized_prediction = model.predict(best_config)[0]

    st.success("🚀 Optimized Results")

    st.write("Optimized Hardness:", round(optimized_prediction[0], 2))
    st.write("Optimized Dissolution:", round(optimized_prediction[1], 2))
    st.write("Optimized Uniformity:", round(optimized_prediction[2], 2))
    st.write("Optimized Carbon:", round(optimized_prediction[3], 2))
    st.write("Optimization Score:", round(best_score, 2))