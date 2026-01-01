import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def run_backtest(ticker="EURUSD=X", days_to_test=60):
    print(f"--- Iniciando Backtesting para {ticker} ---")
    
    # 1. Descarga de datos
    data = yf.download(ticker, period="2y", interval="1d")
    if data.empty: return
    
    # 2. Preparación de indicadores (Igual que en la App)
    df = data.copy()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    features = ['Close', 'SMA_100', 'SMA_200', 'RSI']
    df_filtered = df[features].bfill().dropna()
    
    # 3. Escalado
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_filtered.values)
    
    # 4. Dividir datos: Entrenamiento (todo excepto los últimos X días) y Test
    train_size = len(scaled_data) - days_to_test
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:] # Ventana de 60 para la primera predicción
    
    # Preparar X_train, y_train
    window = 60
    X_train, y_train = [], []
    for i in range(window, len(train_data)):
        X_train.append(train_data[i-window:i, :])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 5. Entrenar modelo rápido para el test
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    print("Entrenando modelo de prueba...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # 6. Realizar predicciones sobre el periodo de Test
    X_test = []
    for i in range(window, len(test_data)):
        X_test.append(test_data[i-window:i, :])
    X_test = np.array(X_test)
    
    predictions_scaled = model.predict(X_test)
    
    # Des-escalar
    dummy = np.zeros((len(predictions_scaled), 4))
    dummy[:, 0] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    # Precios Reales
    actual_prices = df_filtered['Close'].values[-days_to_test:]
    
    # 7. Resultados y Gráfica
    rmse = np.sqrt(np.mean((predictions - actual_prices)**2))
    print(f"✅ Backtesting completado. Error Medio (RMSE): {rmse:.4f}")
    
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices, label="Precio Real", color="blue")
    plt.plot(predictions, label="Predicción IA", color="orange", linestyle="--")
    plt.title(f"Backtesting StockAI: {ticker}")
    plt.legend()
    plt.savefig('backtest_result.png')
    print("📈 Gráfica guardada como 'backtest_result.png'")

if __name__ == "__main__":
    run_backtest()