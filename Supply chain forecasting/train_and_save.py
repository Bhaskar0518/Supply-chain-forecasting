import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load dataset
file_path = r'C:\Users\vbhas\OneDrive\Desktop\New Foldercc\Project for 3 unifi raw materials\supply_chain_data.csv'
data = pd.read_csv(file_path).dropna()

# Drop irrelevant columns
drop_cols = ['SKU', 'Supplier name', 'Location', 'Routes',
             'Shipping carriers', 'Inspection results', 'Customer demographics']
data = data.drop(columns=[col for col in drop_cols if col in data.columns])

# One-hot encode categorical columns
categorical_cols = ['Product type', 'Transportation modes']
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col in data.columns])

# Features/target
target = 'Revenue generated'
X = data.drop(columns=[target])
y = data[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2)

# Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"📉 Test MSE: {mse:.2f}")

# Save model + scaler + history
model.save("demand_forecasting_model.keras")
pd.to_pickle(scaler, "scaler.pkl")
pd.DataFrame(history.history).to_csv("training_history.csv", index=False)
print("✅ Model, scaler, and history saved in the same folder.")
