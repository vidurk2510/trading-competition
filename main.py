import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import kagglehub
# import zipfile  <-- We don't need this anymore
import os

# --- Step 1: Load the Pipeline ---
print("Loading pipeline on CPU...")
pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-tiny",
  device_map="cpu",
  torch_dtype=torch.float32,
)
print("✅ Pipeline loaded successfully!")

# --- Step 2: Load Your Kaggle Data ---
print("Loading Kaggle dataset...")
# This function downloads AND unzips the data,
# returning the path to the FOLDER where the CSVs are.
dataset_folder_path = kagglehub.dataset_download("sudalairajkumar/cryptocurrencypricehistory")

# We don't need to unzip. Just join the folder path with the file name.
csv_path = os.path.join(dataset_folder_path, "coin_Bitcoin.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Could not find {csv_path}")
    print("Maybe try 'coin_Ethereum.csv' or another coin name?")
    exit()

print(f"✅ Loaded {csv_path}. Total rows: {len(df)}")


# --- Step 3: Prepare Data for Chronos ---
# Give the model the last 200 days of "Close" prices as context
context_data = df['Close'].values[-200:]
context_tensor = torch.tensor(context_data, dtype=torch.float32)

# --- Step 4: Get the Forecast ---
# Predict the next 7 days
prediction_length = 7

print(f"Forecasting next {prediction_length} days...")
forecast_samples = pipeline.predict(
    context_tensor,
    prediction_length
)

# Get the median (50th percentile) forecast
median_forecast = torch.quantile(forecast_samples[0], 0.5, dim=0).numpy()

print(f"Model forecasts the next {prediction_length} prices as: {median_forecast}")

# --- Step 5: Run Your "Buy/Sell" Bot Logic ---
current_price = context_data[-1]
predicted_final_price = median_forecast[-1]

# Your strategy: "buy if the price is predicted to rise 3%"
buy_threshold = 1.03  # 3% increase
sell_threshold = 0.97 # 3% decrease

print("--- Bot Decision Logic ---")
print(f"Current Price:   ${current_price:,.2f}")
print(f"Forecasted Price: ${predicted_final_price:,.2f}")

if predicted_final_price > (current_price * buy_threshold):
    print("DECISION: 📈 BUY")

elif predicted_final_price < (current_price * sell_threshold):
    print("DECISION: 📉 SELL")

else:
    print("DECISION: 😐 HOLD")