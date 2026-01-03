📈 StockAI V3 Pro: Adaptive Multi-Indicator Intelligence
StockAI V3 Pro is a financial predictive analysis platform that merges Deep Learning with traditional technical analysis. It utilizes a Long Short-Term Memory (LSTM) Recurrent Neural Network architecture to process time-series data and project price trends in global financial markets.

🔗 Quick Links

Web Deployment: https://stockai-predictor-rasosa.streamlit.app

Project Status: Production / Stable (Python 3.11).

📝 1. Problem Description
Predicting financial markets is a complex challenge due to high volatility and the non-linear nature of data. Many traditional models fail because they ignore momentum (RSI) or long-term trends (Moving Averages).

StockAI V3 Pro solves this through:

Multivariate Analysis: The model integrates the Relative Strength Index (RSI) and Simple Moving Averages (SMA 100/200) as input features for the neural network.

Dynamic Training: The app trains a neural network in real-time using the latest data from Yahoo Finance, adapting to current market conditions.

⚙️ 2. Internal Mechanics and Structure
AI Architecture

The "brain" of the app is an LSTM network designed to remember long-term historical patterns.

Extraction: Downloads data via yfinance.

Feature Engineering: Calculates technical indicators (RSI, SMA) in real-time.

Scaling: Normalizes data using MinMaxScaler for optimal learning.

Prediction: Projects the value for the next period based on a sliding observation window.

StockAI-Predictor-V2/
├── .python-version          # Forces Python 3.11 for cloud stability
├── requirements.txt         # Core dependencies (TensorFlow-CPU, Streamlit, etc.)
├── app/
│   └── app.py               # Main Entry point for the Streamlit Dashboard
├── src/                     # Source code logic
│   ├── model.py             # LSTM Architecture
│   ├── data_downloader.py   # Yahoo Finance API integration
│   ├── strategy.py          # Technical indicators calculation
│   ├── train.py             # Model training and scaling logic
│   └── backtesting.py       # Validation and RMSE calculation engine
├── models/                  # Saved .h5 models and scalers
└── notebooks/               # EDA (Exploratory Data Analysis) and research


🚀 3. Local Execution Guide
Prerequisites

Python 3.11 installed.

Active internet connection.

Installation Steps

Clone the Project:

Bash
git clone https://github.com/Rasosa31/StockAI-Predictor-V2.git
cd StockAI-Predictor-V2
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the Application:

Bash
streamlit run app/app.py
📊 4. Technical Findings and EDA
Metric Reliability: The model uses RMSE (Root Mean Square Error). A low RMSE indicates the AI tracked the actual price trends closely during backtesting.

Optimization: The system was optimized for Streamlit Cloud by using tensorflow-cpu to manage memory constraints effectively.

🚀 5. Future Improvements (Roadmap)
MetaTrader 5 (MT5) Integration: Development of a bridge script for automated execution in Forex (FX) accounts.

Sentiment Module: Using NLP to include news sentiment as an additional input variable.

Developed by Ramiro Sosa - Capstone Final Project