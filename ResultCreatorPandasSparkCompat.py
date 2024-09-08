import os
from datetime import timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

def read_parquet_files(input_path: str) -> pd.DataFrame:
    return pd.read_parquet(input_path)

def calculate_metrics(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    grouped = filtered_df.groupby('branch')
    
    metrics = grouped.agg({
        'trade_returns_day': lambda x: (np.exp(np.sum(np.log(x))) - 1) * 100,
        'date': 'count'
    }).rename(columns={'trade_returns_day': 'cum_return', 'date': 'days'})
    
    metrics['years'] = metrics['days'] / 252
    metrics['cagr'] = (np.power(1 + metrics['cum_return'] / 100, 1 / metrics['years']) - 1) * 100
    
    # Calculate max drawdown
    cumulative_returns = grouped['trade_returns_day'].cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['max_drawdown'] = grouped.apply(lambda x: drawdown.loc[x.index].min()).abs() * 100
    
    # Calculate percent days in market
    metrics['percent_days_in_market'] = grouped['trade_returns_day'].apply(lambda x: (x != 1).sum() / len(x) * 100)
    
    # Calculate ROMAD
    days_diff = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    metrics['romad'] = metrics['cagr'] / metrics['max_drawdown'] * np.sqrt(252 / days_diff)
    
    return metrics.reset_index()

def select_top_branches(df: pd.DataFrame, branch_count: int, min_return: float) -> pd.DataFrame:
    return df[df['cagr'] >= min_return].sort_values('romad', ascending=False).head(branch_count)

def process_date_range(args):
    current_date, df, look_back_window, branch_count, min_return, look_forward_window = args
    
    back_start_date = (current_date - timedelta(days=look_back_window)).replace(day=1)
    back_end_date = current_date - timedelta(days=1)
    forward_end_date = current_date + timedelta(days=look_forward_window)
    
    # Calculate metrics for look-back period
    back_metrics = calculate_metrics(df, back_start_date.strftime("%Y-%m-%d"), back_end_date.strftime("%Y-%m-%d"))
    top_branches = select_top_branches(back_metrics, branch_count, min_return)
    
    # Calculate metrics for forward-looking period
    forward_metrics = calculate_metrics(df, current_date.strftime("%Y-%m-%d"), forward_end_date.strftime("%Y-%m-%d"))
    
    # Join back and forward metrics
    combined_metrics = pd.merge(top_branches[['branch', 'cagr', 'romad']], 
                                forward_metrics[['branch', 'cagr', 'romad']], 
                                on='branch', suffixes=('_back', '_forward'))
    
    # Collect results
    period_results = {
        "period": current_date.strftime("%Y-%m"),
        "avg_back_cagr": combined_metrics['cagr_back'].mean(),
        "avg_forward_cagr": combined_metrics['cagr_forward'].mean()
    }
    
    # Save detailed results
    combined_metrics.to_parquet(f"results-pandas/metrics_{current_date.strftime('%Y_%m')}.parquet")
    
    return period_results

def main() -> None:
    # Parameters
    look_back_window = 90
    branch_count = 100
    min_return = 10
    look_forward_window = 30
    
    # Read all Parquet files
    df = read_parquet_files("output_data_spark")
    
    # Generate date range
    start_date = "2000-01-01"
    end_date = "2023-12-31"
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Prepare arguments for parallel processing
    args_list = [(date, df, look_back_window, branch_count, min_return, look_forward_window) for date in date_range]
    
    # Use multiprocessing to parallelize the computation
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_date_range, args_list)
    
    # Save summary results
    pd.DataFrame(results).to_csv("results-pandas/monthly_returns.csv", index=False)

if __name__ == "__main__":
    if not os.path.exists("results-pandas"):
        os.makedirs("results-pandas")
    main()
