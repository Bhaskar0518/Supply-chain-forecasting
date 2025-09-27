import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load model, scaler, and history
# -----------------------------
model = keras.models.load_model("demand_forecasting_model.keras")
scaler = pd.read_pickle("scaler.pkl")
history = pd.read_csv("training_history.csv")

# Reload dataset for template
file_path = r'C:\Users\vbhas\OneDrive\Desktop\New Foldercc\Project for 3 unifi raw materials\supply_chain_data.csv'
data = pd.read_csv(file_path).dropna()
drop_cols = ['SKU', 'Supplier name', 'Location', 'Routes',
             'Shipping carriers', 'Inspection results', 'Customer demographics']
data = data.drop(columns=[col for col in drop_cols if col in data.columns])
data = pd.get_dummies(data, columns=[col for col in ['Product type', 'Transportation modes'] if col in data.columns])

X = data.drop(columns=['Revenue generated'])
y = data['Revenue generated']

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Supply Chain Demand Forecasting Dashboard")

price = st.number_input("Price", min_value=0.0, value=1.0)
availability = st.number_input("Availability", min_value=0.0, value=1.0)
lead_time = st.number_input("Lead Time", min_value=0.0, value=1.0)

product_type_options = [col.replace("Product type_", "") for col in X.columns if col.startswith("Product type_")]
transport_options = [col.replace("Transportation modes_", "") for col in X.columns if col.startswith("Transportation modes_")]

product_type = st.selectbox("Product Type", product_type_options if product_type_options else ["N/A"])
transport_mode = st.selectbox("Transportation Mode", transport_options if transport_options else ["N/A"])

# Build input row
template = X.iloc[0].copy()
template[:] = 0
if "Price" in template: template["Price"] = price
if "Availability" in template: template["Availability"] = availability
if "Lead times" in template: template["Lead times"] = lead_time
if f"Product type_{product_type}" in template: template[f"Product type_{product_type}"] = 1
if f"Transportation modes_{transport_mode}" in template: template[f"Transportation modes_{transport_mode}"] = 1

new_data_df = pd.DataFrame([template])
new_data_scaled = scaler.transform(new_data_df)

# Prediction
if st.button("Predict Revenue"):
    predicted_revenue = model.predict(new_data_scaled)[0][0]
    st.metric("📈 Predicted Revenue", f"{predicted_revenue:,.2f}")

# Charts
st.subheader("Model Evaluation")

# Chart 1: Training vs Validation Loss
fig1, ax1 = plt.subplots()
ax1.plot(history["loss"], label="Train Loss")
ax1.plot(history["val_loss"], label="Validation Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Mean Squared Error")
ax1.set_title("Training vs Validation Loss")
ax1.legend()
st.pyplot(fig1)

# Chart 2: True vs Predicted Revenue
y_pred = model.predict(scaler.transform(X))
fig2, ax2 = plt.subplots()
ax2.scatter(y, y_pred, alpha=0.6)
ax2.set_xlabel("True Revenue")
ax2.set_ylabel("Predicted Revenue")
ax2.set_title("True vs Predicted Revenue")
st.pyplot(fig2)

# Chart 3: Revenue Distribution
fig3, ax3 = plt.subplots()
ax3.hist(y, bins=30, color="skyblue", edgecolor="black")
ax3.set_xlabel("Revenue")
ax3.set_ylabel("Frequency")
ax3.set_title("Revenue Distribution")
st.pyplot(fig3)

# Show MSE
mse = mean_squared_error(y, y_pred)
st.write(f"📉 **Test MSE:** {mse:.2f}")
