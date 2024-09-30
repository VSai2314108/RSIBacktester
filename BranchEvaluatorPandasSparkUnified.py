import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import multiprocessing as mp

# Remove unused imports
# from datetime import datetime, timedelta

def load_data(ticker):
    file_path = f'./indicator_data/{ticker}.csv'
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    return df

def evaluate_condition(df, indicator, direction, value, days):
    indicator_values = df[indicator]
    
    if direction == '>':
        condition = indicator_values > value
    else:
        condition = indicator_values < value
    
    # Change the rolling operation
    return condition.rolling(window=days).min() == 1

def evaluate_branch(branch):
    parts = branch.split('-')
    indicator, period, indicator_ticker, direction, threshold, days, trading_ticker = parts
    
    indicator_df = load_data(indicator_ticker)
    trading_df = load_data(trading_ticker)
    
    common_dates = indicator_df.index.intersection(trading_df.index)
    indicator_df = indicator_df.loc[common_dates]
    trading_df = trading_df.loc[common_dates]
    
    condition_met = evaluate_condition(indicator_df, f"{indicator}_{period}", direction, float(threshold), int(days))
    
    result = pd.DataFrame(index=trading_df.index)
    result['condition_met'] = condition_met.astype(int)
    
    # Shift condition_met forward by one day and calculate trade returns
    result['shifted_condition'] = result['condition_met'].shift(1)
    result['trade_returns_day'] = (result['shifted_condition'] * trading_df['close'].pct_change()) + 1
    
    return result[['condition_met', 'trade_returns_day']].to_dict(orient='index')

def append_to_parquet(data, branch, output_file):
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.columns = ['date', 'condition_met', 'trade_returns_day']
    df['date'] = pd.to_datetime(df['date'])
    
    # Add year and month columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Add branch column
    df['branch'] = branch
    
    # Convert pandas DataFrame to PyArrow Table
    new_table = pa.Table.from_pandas(df)
    
    if os.path.exists(output_file):
        # If the file exists, read its schema
        existing_schema = pq.read_schema(output_file)
        # Ensure the new table matches the existing schema
        new_table = new_table.cast(existing_schema)
        
        # Open the existing file in append mode
        with pq.ParquetWriter(output_file, existing_schema, append=True) as writer:
            writer.write_table(new_table)
    else:
        # If the file doesn't exist, create it
        pq.write_table(new_table, output_file)

def process_branch(args):
    branch, output_file = args
    try:
        result = evaluate_branch(branch)
        append_to_parquet(result, branch, output_file)
        return None
    except Exception as e:
        return f"Error processing branch {branch}: {str(e)}"

def main():
    with open('branches.txt', 'r') as f:
        branches = f.read().splitlines()
    
    output_file = './output_data_spark/unified_output.parquet'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Determine the number of CPU cores to use
    num_cores = mp.cpu_count() - 4  # Leave one core free
    
    # Create a pool of worker processes
    with mp.Pool(num_cores) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap_unordered(process_branch, [(branch, output_file) for branch in branches]),
            total=len(branches),
            desc="Processing branches",
            unit="branch"
        ))
    
    # Print any errors that occurred during processing
    errors = [error for error in results if error is not None]
    for error in errors:
        print(error)
    
    print("Branch evaluation completed.")

if __name__ == "__main__":
    main()
