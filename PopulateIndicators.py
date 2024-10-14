import pandas as pd
import numpy as np
import os
import ta  # Make sure to import the ta library

def calculate_rsi_vectorized(data: pd.Series, periods):
    def calculate_rsi(period):
        rsi_series = ta.momentum.RSIIndicator(data, window=period).rsi()
        rsi_series[:period-1] = np.nan  # Set RSI values to NaN if there's not enough data
        rsi_series = rsi_series.round(1)  # Round to the nearest tenth
        return rsi_series
    
    rsi_dict = {f'rsi_{period}': calculate_rsi(period) for period in periods}
    return pd.DataFrame(rsi_dict)

def process_file(file_path):
    df = pd.read_csv(file_path)
    df.set_index('date', inplace=True)
    df['pct_change'] = df['close'].pct_change()
    
    periods = range(2, 201)
    rsi_df = calculate_rsi_vectorized(df['close'], periods)
    
    return pd.concat([df, rsi_df], axis=1)

def main():
    output_dir = './indicator_data'
    data_dir = './data'
    os.makedirs(output_dir, exist_ok=True)
    
    # read the tickers from the tickers-ind.txt
    with open('./tickers-ind.txt', 'r') as file:
        tickers = [line.strip() for line in file]
    
    for ticker in tickers:
        print(f"Processing {ticker}")
        file_path = os.path.join(data_dir, f"{ticker}.csv")
        df = process_file(file_path)
            
        output_file = os.path.join(output_dir, f"{ticker.upper()}.csv")
        df.to_csv(output_file)
        print(f"Saved {output_file}")

if __name__ == "__main__":
    main()
