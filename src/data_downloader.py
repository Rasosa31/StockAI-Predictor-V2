import yfinance as yf
import pandas as pd
import os

def download_multitoken_data():
    # Lista de 20 activos diversificados (Tecnología, Índices, Cripto, Oro)
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',  # Big Tech
        'SPY', 'QQQ', 'DIA',                                     # Índices
        'BTC-USD', 'ETH-USD',                                    # Cripto
        'AMD', 'INTC', 'PYPL', 'NFLX', 'ADBE',                   # Crecimiento
        'GLD', 'VTI', 'TLT'                                      # Refugios/Bonos
    ]
    
    all_data = []
    print(f"Iniciando descarga de {len(tickers)} activos...")

    for ticker in tickers:
        try:
            print(f"Descargando {ticker}...")
            # Descargamos 5 años de datos diarios
            df = yf.download(ticker, period="5y", interval="1d")
            if not df.empty:
                df['Ticker'] = ticker  # Identificador para el modelo
                all_data.append(df)
        except Exception as e:
            print(f"Error con {ticker}: {e}")

    # Combinamos todo en un solo DataFrame
    final_df = pd.concat(all_data)
    
    # Crear carpeta data si no existe
    os.makedirs('data', exist_ok=True)
    
    # Guardar el dataset maestro
    final_df.to_csv('data/multi_stock_data.csv')
    print("\n✅ Proceso completado. Archivo guardado en: data/multi_stock_data.csv")
    print(f"Total de registros descargados: {len(final_df)}")

if __name__ == "__main__":
    download_multitoken_data()