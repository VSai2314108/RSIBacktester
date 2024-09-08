import pandas as pd
import os

def calculate_rsi_vectorized(data: pd.Series, periods):
    delta = data.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    avg_gains = [gains.rolling(window=period).mean() for period in periods]
    avg_losses = [losses.rolling(window=period).mean() for period in periods]
    
    rs_values = [avg_gain / avg_loss for avg_gain, avg_loss in zip(avg_gains, avg_losses)]
    rsi_values = [100 - (100 / (1 + rs)) for rs in rs_values]
    
    return pd.DataFrame({f'rsi_{period}': rsi for period, rsi in zip(periods, rsi_values)})

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
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv') and file != 'last_updated.csv':
            ticker = os.path.splitext(file)[0]
            print(f"Processing {ticker}")
            
            file_path = os.path.join(data_dir, file)
            df = process_file(file_path)
            
            output_file = os.path.join(output_dir, f"{ticker}.csv")
            df.to_csv(output_file)
            print(f"Saved {output_file}")

if __name__ == "__main__":
    main()
