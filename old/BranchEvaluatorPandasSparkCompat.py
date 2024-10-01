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

# Update the save_as_parquet function to handle the new column
def save_as_parquet(data, branch, output_dir):
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.columns = ['date', 'condition_met', 'trade_returns_day']
    df['date'] = pd.to_datetime(df['date'])
    
    # Add year and month columns for partitioning
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Add branch column
    df['branch'] = branch
    
    # Convert pandas DataFrame to PyArrow Table
    output_table = pa.Table.from_pandas(df)
    
    # Create partitions based on year and month, overwriting existing data
    pq.write_to_dataset(
        output_table,
        root_path=output_dir,
        partition_cols=['year', 'month', 'branch'],
        existing_data_behavior='delete_matching'
    )

def process_branch(branch):
    try:
        result = evaluate_branch(branch)
        output_dir = './output_data_spark'
        save_as_parquet(result, branch, output_dir)
        return None
    except Exception as e:
        return f"Error processing branch {branch}: {str(e)}"

def main():
    with open('branches.txt', 'r') as f:
        branches = f.read().splitlines()
    
    output_dir = './output_data_spark'
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the number of CPU cores to use
    num_cores = mp.cpu_count() - 4  # Leave one core free
    
    # Create a pool of worker processes
    with mp.Pool(num_cores) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap_unordered(process_branch, branches),
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
