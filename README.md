#  Supply Chain Demand Forecasting using Deep Learning

An end-to-end machine learning pipeline for predicting revenue in a supply chain context. This project covers data preprocessing, model training, evaluation, and deployment via Streamlit — designed for real-world business impact and portfolio presentation.

---

##  Highlights

- ✅ Cleaned and preprocessed supply chain data
- ✅ Trained a deep neural network using TensorFlow/Keras
- ✅ Achieved strong performance (Test MSE: **123456.78**) *(replace with your actual value)*
- ✅ Saved model and scaler for future inference
- ✅ Visualized training history and predictions
- ✅ Deployed an interactive dashboard using Streamlit

---

##  Tech Stack

- Python, pandas, NumPy  
- scikit-learn, TensorFlow/Keras  
- Matplotlib  
- Streamlit

---

##  Repository Structure

File

Description

train_and_save.py

Training script with model saving

app.py

Streamlit dashboard for interactive inference

demand_forecasting_model.keras

Saved Keras model

scaler.pkl

Saved StandardScaler for preprocessing

training_history.csv

Epoch-wise training and validation loss

supply_chain_data.csv

Cleaned dataset used for training

report.pdf

Formal project report

requirements.txt

Python dependencies

README.md

Project overview and instructions



🎛️ How to Run

Clone the repo:

git clone https://github.com/Bhaskar0518/Supply-chain-forecasting.git
cd Supply-chain-forecasting

Install dependencies:

pip install -r requirements.txt

Run the dashboard:

streamlit run app.py

*** Report

Read the full report here:📄 Supply Chain Forecasting Report

*** Future Improvements

Add time-series modeling (e.g., LSTM)

Integrate real-time API data

Add model explainability (SHAP, feature importance)

Automate retraining pipeline



Bhaskar — GitHub Profile
