import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os

# 1. Page Configuration
st.set_page_config(page_title="StockAI V3 Pro - Elite Edition", layout="wide")

def train_multivariate_model(X, y):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=12, batch_size=32, verbose=0)
    return model

# 2. Interface and Sidebar
st.title("🤖 StockAI V3: Multi-Indicator Intelligence")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    ticker = st.text_input(
        "Enter Stock Ticker:", 
        value="AAPL", 
        help="Important: Use only Yahoo Finance symbols (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD', 'GC=F')."
    ).upper()
    st.caption("⚠️ Make sure the ticker exists on finance.yahoo.com")

    interval_label = st.selectbox("Select Timeframe:", ["Daily", "Weekly", "Monthly"])
    
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    interval_code = interval_map[interval_label]
    
    st.divider()
    st.header("🛠️ Extra Tools")
    show_backtest = st.checkbox("Enable Backtesting Analysis")

# --- 3. Adaptive Data Engine ---
data = yf.download(ticker, period="max", interval=interval_code)
if isinstance(data.columns, pd.MultiIndex): 
    data.columns = data.columns.get_level_values(0)

if not data.empty and len(data) > 30:
    df = data.copy()
    total_candles = len(df)
    
    # Dynamic SMA calculation based on asset history
    w100 = 100 if total_candles >= 100 else max(2, total_candles // 2)
    w200 = 200 if total_candles >= 200 else max(2, total_candles - 1)
    
    df['SMA_100'] = df['Close'].rolling(window=w100).mean()
    df['SMA_200'] = df['Close'].rolling(window=w200).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    features = ['Close', 'SMA_100', 'SMA_200', 'RSI']
    df_filtered = df[features].bfill().dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_filtered.values)

    # 4. Main Prediction
    if st.button(f"🚀 Run {interval_label} Projection"):
        with st.spinner(f"Analyzing {ticker} trends..."):
            window = 60 if len(scaled_data) > 60 else len(scaled_data) // 2
            X, y = [], []
            for i in range(window, len(scaled_data)):
                X.append(scaled_data[i-window:i, :])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            model = train_multivariate_model(X, y)
            
            last_window = scaled_data[-window:].reshape(1, window, len(features))
            pred_scaled = model.predict(last_window)
            dummy = np.zeros((1, len(features)))
            dummy[0, 0] = pred_scaled[0][0]
            pred_final = float(scaler.inverse_transform(dummy)[0][0])
            
            current_price = float(df_filtered['Close'].iloc[-1])
            diff = pred_final - current_price
            pct = (diff / current_price) * 100
            precision = "4f" if current_price < 2 else "2f"
            
            # Metrics Presentation
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${current_price:,.{precision}}")
            m2.metric(f"AI Prediction", f"${pred_final:,.{precision}}", f"{diff:,.{precision}}")
            m3.metric("Expected Movement", f"{pct:.2f}%")
            
            # Main Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered.index[-120:], y=df_filtered['Close'][-120:], name="Historical Price"))
            
            # Prediction line
            delta_time = pd.Timedelta(days=1 if interval_code=="1d" else 7 if interval_code=="1wk" else 30)
            fig.add_trace(go.Scatter(
                x=[df_filtered.index[-1], df_filtered.index[-1] + delta_time], 
                y=[current_price, pred_final], 
                name="AI Future Path", 
                line=dict(color='orange', dash='dash', width=3)
            ))
            
            fig.update_layout(template="plotly_dark", title=f"{ticker} Performance Analysis ({interval_label})")
            st.plotly_chart(fig, use_container_width=True)

    # 5. Backtesting Analysis
    if show_backtest:
        st.divider()
        st.subheader(f"📊 Historical Backtesting ({interval_label})")
        if st.button("🔄 Run Accuracy Test"):
            with st.spinner("Calculating historical accuracy..."):
                test_days = 30
                window = 60 if len(scaled_data) > 60 else 10
                
                # Test Training
                X_b, y_b = [], []
                train_data_b = scaled_data[:-test_days]
                for i in range(window, len(train_data_b)):
                    X_b.append(train_data_b[i-window:i, :])
                    y_b.append(train_data_b[i, 0])
                
                model_b = train_multivariate_model(np.array(X_b), np.array(y_b))
                
                # Test Segment
                X_test = []
                test_segment = scaled_data[-(test_days + window):]
                for i in range(window, len(test_segment)):
                    X_test.append(test_segment[i-window:i, :])
                
                preds_b_scaled = model_b.predict(np.array(X_test))
                dummy_b = np.zeros((len(preds_b_scaled), 4))
                dummy_b[:, 0] = preds_b_scaled.flatten()
                preds_b = scaler.inverse_transform(dummy_b)[:, 0]
                real_b = df_filtered['Close'].values[-test_days:]
                
                rmse = np.sqrt(np.mean((preds_b - real_b)**2))
                st.info(f"Model Performance Score: RMSE = {rmse:.4f}")
                
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(y=real_b, name="Actual Data", line=dict(color='blue')))
                fig_b.add_trace(go.Scatter(y=preds_b, name="AI Prediction", line=dict(color='orange', dash='dot')))
                fig_b.update_layout(template="plotly_dark", height=350, title="Backtesting: Reality vs Prediction")
                st.plotly_chart(fig_b, use_container_width=True)
else:
    st.error("Insufficient historical data to generate a projection for this asset.")