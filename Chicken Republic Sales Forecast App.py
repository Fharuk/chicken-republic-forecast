import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import os

# -------------------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Chicken Republic Forecast", layout="wide")
DATA_FILE = "chicken_republic_lagos_sales (1).xlsx"

# -------------------------------------------------------------------------------------------------
# CACHED FUNCTIONS (The Performance Fix)
# -------------------------------------------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    """Loads data once and keeps it in memory."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_excel(filepath)
        # Ensure Date is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

@st.cache_resource
def train_models(df):
    """Trains models once and caches them. Re-runs only if df changes."""
    quantiles = [0.1, 0.5, 0.9]
    feature_cols = [
        "Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "IsWeekend",
        "qty_lag_1", "qty_lag_7", "roll_mean_7", "roll_std_7", "price_delta_7",
        "Location_enc", "Product Category_enc", "Product_enc"
    ]
    
    model_dict = {}
    
    # Feature Engineering for Training
    df_feat = df.copy()
    df_feat["Year"] = df_feat["Date"].dt.year
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Day"] = df_feat["Date"].dt.day
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["WeekOfYear"] = df_feat["Date"].dt.isocalendar().week
    df_feat["IsWeekend"] = df_feat["DayOfWeek"].isin([5,6]).astype(int)
    
    # Lag features
    df_feat["qty_lag_1"] = df_feat["Quantity Sold"].shift(1).fillna(0)
    df_feat["qty_lag_7"] = df_feat["Quantity Sold"].shift(7).fillna(0)
    df_feat["roll_mean_7"] = df_feat["Quantity Sold"].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    df_feat["roll_std_7"] = df_feat["Quantity Sold"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    
    # Price delta
    # Check if 'Unit Price (NGN)' exists, handle potential missing column
    price_col = "Unit Price (NGN)" if "Unit Price (NGN)" in df_feat.columns else df_feat.columns[2] # Fallback
    df_feat["price_delta_7"] = df_feat[price_col] - df_feat[price_col].shift(1).rolling(7, min_periods=1).mean().fillna(0)

    # Encoders (We need to save these to use them in prediction)
    encoders = {}
    for col in ["Location", "Product Category", "Product"]:
        le = LabelEncoder()
        df_feat[col + "_enc"] = le.fit_transform(df_feat[col].astype(str))
        encoders[col] = le

    # Train
    X_train = df_feat[feature_cols].fillna(0)
    y_train = df_feat["Quantity Sold"]

    for q in quantiles:
        lgb_q = LGBMRegressor(
            objective='quantile', alpha=q, 
            learning_rate=0.1, num_leaves=31, 
            n_estimators=200, random_state=42
        )
        lgb_q.fit(X_train, y_train)
        model_dict[q] = lgb_q
        
    return model_dict, encoders, feature_cols

# -------------------------------------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------------------------------------
def main():
    st.title("🍗 Chicken Republic Sales Forecasting")

    # 1. Load Data
    df = load_data(DATA_FILE)
    if df is None:
        st.error(f"🚨 CRITICAL ERROR: Could not find '{DATA_FILE}'. Please upload this file to your GitHub repository.")
        st.stop()

    # 2. Train Models (Cached)
    with st.spinner("Calibrating Forecasting Models..."):
        model_dict, encoders, feature_cols = train_models(df)

    # 3. Sidebar Inputs
    product = st.sidebar.selectbox("Select Product", sorted(df["Product"].unique()))
    location = st.sidebar.selectbox("Select Location", sorted(df["Location"].unique()))

    # 4. Filter Data
    df_sku = df[(df["Product"] == product) & (df["Location"] == location)].sort_values("Date").reset_index(drop=True)
    
    if df_sku.empty:
        st.warning("No data found for this combination.")
        st.stop()

    # 5. Prepare Prediction Row
    # We construct the features for the "Next Day"
    last_real_date = df_sku["Date"].iloc[-1]
    next_date = last_real_date + pd.Timedelta(days=1)
    
    last_row = pd.DataFrame(index=[0])
    last_row["Year"] = next_date.year
    last_row["Month"] = next_date.month
    last_row["Day"] = next_date.day
    last_row["DayOfWeek"] = next_date.dayofweek
    last_row["WeekOfYear"] = next_date.isocalendar().week
    last_row["IsWeekend"] = 1 if next_date.dayofweek in [5,6] else 0
    
    # Feature calculation (Simplified logic for the 'next step')
    # Using the last known values for lags
    last_row["qty_lag_1"] = df_sku["Quantity Sold"].iloc[-1]
    last_row["qty_lag_7"] = df_sku["Quantity Sold"].iloc[-7] if len(df_sku) >= 7 else df_sku["Quantity Sold"].iloc[-1]
    
    # Rolling stats (approximate using tail)
    last_7_days = df_sku["Quantity Sold"].tail(7)
    last_row["roll_mean_7"] = last_7_days.mean()
    last_row["roll_std_7"] = last_7_days.std() if len(last_7_days) > 1 else 0
    
    # Price Delta (Assume price stays same as last day)
    price_col = "Unit Price (NGN)"
    current_price = df_sku[price_col].iloc[-1]
    avg_past_price = df_sku[price_col].tail(7).mean()
    last_row["price_delta_7"] = current_price - avg_past_price

    # Apply Encoders
    try:
        last_row["Location_enc"] = encoders["Location"].transform([location])[0]
        last_row["Product Category_enc"] = encoders["Product Category"].transform([df_sku["Product Category"].iloc[0]])[0]
        last_row["Product_enc"] = encoders["Product"].transform([product])[0]
    except ValueError:
        st.error("Encoding Error: Selected item contains unseen labels.")
        st.stop()

    # 6. Predict
    X_new = last_row[feature_cols].fillna(0)
    forecast = {f"P{int(q*100)}": model.predict(X_new)[0] for q, model in model_dict.items()}

    # 7. Visualization
    st.subheader(f"Forecast for {next_date.date()}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("P10 (Worst Case)", f"{forecast['P10']:.0f}")
    col2.metric("P50 (Expected)", f"{forecast['P50']:.0f}")
    col3.metric("P90 (Best Case)", f"{forecast['P90']:.0f}")

    st.subheader("Historical Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot last 60 days for clarity
    plot_df = df_sku.tail(60)
    ax.plot(plot_df["Date"], plot_df["Quantity Sold"], marker='o', label="Actual Sales", alpha=0.7)
    ax.axhline(y=forecast["P50"], color='red', linestyle='--', label="Forecast P50")
    ax.set_title(f"Sales Trend: {product}")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()