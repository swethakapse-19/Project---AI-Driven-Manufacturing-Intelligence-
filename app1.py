import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

process_df = pd.read_excel("Hackathon/_h_batch_process_data.xlsx")
production_df = pd.read_excel("Hackathon/_h_batch_production_data.xlsx")

# Ensure same datatype for merge
process_df["Batch_ID"] = process_df["Batch_ID"].astype(str)
production_df["Batch_ID"] = production_df["Batch_ID"].astype(str)

# Merge
df = pd.merge(process_df, production_df, on="Batch_ID")

print("Final Dataset Shape:", df.shape)
df.head()

EMISSION_FACTOR = 0.82  # kg CO2 per kWh (industrial reference)

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
y = targets

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200))
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

correlation_matrix = df[[
    "Power_Consumption_kW",
    "Vibration_mm_s",
    "Motor_Speed_RPM",
    "Temperature_C"
]].corr()

print(correlation_matrix)

plt.figure()
plt.plot(df["Time_Minutes"], df["Power_Consumption_kW"])
plt.xlabel("Time (Minutes)")
plt.ylabel("Power Consumption (kW)")
plt.title("Energy Pattern Across Batch")
plt.show()

energy_mean = df["Power_Consumption_kW"].mean()
energy_std = df["Power_Consumption_kW"].std()

threshold = energy_mean + 2 * energy_std

df["Energy_Anomaly"] = df["Power_Consumption_kW"] > threshold

print("Number of Energy Anomalies:", df["Energy_Anomaly"].sum())

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

    for i in range(100):
        variation = sample_input + np.random.normal(0, 0.05, sample_input.shape)
        prediction = model.predict(variation)[0]
        score = calculate_score(prediction)

        if score > best_score:
            best_score = score
            best_config = variation

    return best_config, best_score


sample = X_test[0].reshape(1, -1)
best_config, best_score = optimize_parameters(model, sample)

print("Optimized Score:", best_score)

golden_signature = {
    "best_score": best_score,
    "best_config": best_config
}

print("Golden Signature Stored.")

new_prediction = model.predict(sample)[0]
new_score = calculate_score(new_prediction)

if new_score > golden_signature["best_score"]:
    golden_signature["best_score"] = new_score
    golden_signature["best_config"] = sample
    print("Golden Signature Updated!")

carbon_limit = df["Carbon_Emission"].mean()

df["Carbon_Compliance"] = df["Carbon_Emission"] <= carbon_limit

print("Carbon Compliance Rate:",
      df["Carbon_Compliance"].mean() * 100, "%")

# ---- USER INPUT SECTION ----

print("Enter Process Parameters")

time_input = float(input("Time (Minutes): "))
temp_input = float(input("Temperature (C): "))
pressure_input = float(input("Pressure (Bar): "))
humidity_input = float(input("Humidity (%): "))
speed_input = float(input("Motor Speed (RPM): "))
force_input = float(input("Compression Force (kN): "))
flow_input = float(input("Flow Rate (LPM): "))
power_input = float(input("Power Consumption (kW): "))
vibration_input = float(input("Vibration (mm/s): "))

user_data = np.array([[ 
    time_input,
    temp_input,
    pressure_input,
    humidity_input,
    speed_input,
    force_input,
    flow_input,
    power_input,
    vibration_input
]])

# Scale input
user_scaled = scaler.transform(user_data)

# Predict
prediction = model.predict(user_scaled)[0]

hardness, dissolution, uniformity, carbon = prediction

print("\n--- PREDICTION OUTPUT ---")
print("Predicted Hardness:", hardness)
print("Predicted Dissolution Rate:", dissolution)
print("Predicted Content Uniformity:", uniformity)
print("Predicted Carbon Emission:", carbon)

best_config, best_score = optimize_parameters(model, user_scaled)

optimized_prediction = model.predict(best_config)[0]

print("\n--- OPTIMIZED OUTPUT ---")
print("Optimized Hardness:", optimized_prediction[0])
print("Optimized Dissolution:", optimized_prediction[1])
print("Optimized Uniformity:", optimized_prediction[2])
print("Optimized Carbon:", optimized_prediction[3])
print("Optimization Score:", best_score)

