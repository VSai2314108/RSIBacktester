import pandas as pd
from tqdm import tqdm
import os
import multiprocessing
from functools import partial
from typing import Union
import numpy as np


def load_data(ticker: str) -> pd.DataFrame:
    file_path: str = f'./indicator_data/{ticker}.csv'
    df: pd.DataFrame = pd.read_csv(file_path, index_col='date', parse_dates=True)
    return df

def evaluate_condition(df: pd.DataFrame, indicator: str, direction: str, value: float, days: int) -> pd.Series:
    indicator_values: pd.Series = df[indicator]
    
    if direction == '>':
        condition: pd.Series = indicator_values > value
    else:
        condition: pd.Series = indicator_values < value
    
    return condition.rolling(window=days).min() == 1

def evaluate_branch(branch: str) -> pd.DataFrame:
    parts: list[str] = branch.split('-')
    indicator, period, indicator_ticker, direction, threshold, days, trading_ticker = parts
    
    indicator_df: pd.DataFrame = load_data(indicator_ticker)
    trading_df: pd.DataFrame = load_data(trading_ticker)
    
    common_dates: pd.DatetimeIndex = indicator_df.index.intersection(trading_df.index)
    indicator_df = indicator_df.loc[common_dates]
    trading_df = trading_df.loc[common_dates]
    
    condition_met: pd.Series = evaluate_condition(indicator_df, f"{indicator}_{period}", direction, float(threshold), int(days))
    
    result: pd.DataFrame = pd.DataFrame(index=trading_df.index)
    result['condition_met'] = condition_met.astype(int)
    
    result['shifted_condition'] = result['condition_met'].shift(1)
    result['trade_returns_day'] = (result['shifted_condition'] * trading_df['close'].pct_change()) + 1
    
    return result[['trade_returns_day']]

def compute_monthly_returns(branch: str, df: pd.DataFrame) -> pd.DataFrame:
    def cagr(series: pd.Series) -> float:
        return series.prod() ** (365 / len(series.index)) - 1
    
    def days_in_market_ratio(series: pd.Series) -> float:
        return (series != 1).sum() / len(series)
    
    def calmar(series: pd.Series) -> float:
        cagr = (series.prod() ** (365 / len(series.index)) - 1)
        max_drawdown = max([1 - min(series.cumprod()), 0.000001])
        return cagr / max_drawdown if max_drawdown != 0 else 0
    
    monthly_returns = df.groupby(['year', 'month']).agg({
        branch: [
            ('cagr', cagr),
            ('days_in_market', days_in_market_ratio),
            ('calmar', calmar)
        ]
    })
    
    monthly_returns.columns = monthly_returns.columns.droplevel(0)
    return monthly_returns
        

    
    # days_in_market_ratio is the number of days where trade_returns_day != 1 over total days in the month
    
    monthly_returns = monthly_returns.reset_index()
    monthly_returns['month-year'] = monthly_returns['year'].astype(str) + '-' + monthly_returns['month'].astype(str).str.zfill(2)
    monthly_returns['branch'] = branch
    monthly_returns = monthly_returns.pivot(index='month-year', columns='branch', values='trade_returns_day')
    return monthly_returns

def process_branch(branch: str) -> pd.DataFrame:
    result: pd.DataFrame = evaluate_branch(branch)
    monthly_returns: pd.DataFrame = compute_monthly_returns(branch, result)
    return monthly_returns

def batch_processor(branches: list[str]) -> pd.DataFrame:
    num_cores = multiprocessing.cpu_count()    
    with multiprocessing.Pool(processes=num_cores) as pool:
        monthly_returns_dfs: list[pd.DataFrame] = list(tqdm(
            pool.imap(process_branch, branches),
            total=len(branches),
            desc="Processing branches",
            unit="branch",
            leave=False
        ))
    
    return pd.concat(monthly_returns_dfs, axis=1)

def gather_monthly_returns(directory: str) -> pd.DataFrame:
    monthly_returns_df: pd.DataFrame = pd.DataFrame()
    files: list[str] = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    for file in tqdm(files, desc="Processing files", unit="file"):
        monthly_returns_df = pd.concat([monthly_returns_df, pd.read_parquet(os.path.join(directory, file))], axis=1)
    return monthly_returns_df

def one_back_one_forward(monthly_returns_df: pd.DataFrame, 
                         back_start: Union[str, pd.Timestamp], 
                         back_end: Union[str, pd.Timestamp], 
                         forward_start: Union[str, pd.Timestamp], 
                         forward_end: Union[str, pd.Timestamp], 
                         top_n: int) -> pd.DataFrame:
    # Convert string dates to datetime if necessary
    if isinstance(back_start, str):
        back_start = pd.to_datetime(back_start)
    if isinstance(back_end, str):
        back_end = pd.to_datetime(back_end)
    if isinstance(forward_start, str):
        forward_start = pd.to_datetime(forward_start)
    if isinstance(forward_end, str):
        forward_end = pd.to_datetime(forward_end)
    
    back_returns: pd.DataFrame = monthly_returns_df.loc[back_start:back_end]
    forward_returns: pd.DataFrame = monthly_returns_df.loc[forward_start:forward_end]
    
    # Calculate cumulative returns for the back period
    back_period_returns: pd.DataFrame = back_returns.cumprod()
    top_n_branches: pd.Series = back_period_returns.iloc[-1].nlargest(top_n)
    print(top_n_branches)
    
    branches_to_check: list[str] = top_n_branches.index.tolist()
    print(branches_to_check)
    
    # Calculate cumulative returns for the forward period
    forward_period_returns: pd.DataFrame = forward_returns.cumprod()
    forward_period_returns: pd.Series = forward_period_returns.iloc[-1].loc[branches_to_check]
    
    # Combine the back period and forward period returns
    dataset: pd.DataFrame = pd.concat([top_n_branches, forward_period_returns], axis=1)
    dataset.columns = ['back_period_returns', 'forward_period_returns']
    
    # Add a row for the averages
    dataset.loc['avg'] = ((dataset.mean() - 1) * 100)
    
    # Calculate the portion of year for back and forward periods
    dataset.loc['portion_of_year'] = [(back_end - back_start).days / 365, (forward_end - forward_start).days / 365]
    
    # Calculate CAGR
    dataset.loc['cagr'] = dataset.loc['avg'] / dataset.loc['portion_of_year']
    
    print(dataset)
    
    return dataset

if __name__ == "__main__":
    # os.makedirs('./monthly_returns', exist_ok=True)
    
    # branches: list[str] = []
    # with open('branches.txt', 'r') as file:
    #     branches = [line.strip() for line in file]
        
    # batch_size: int = 100
    # for i in tqdm(range(0, len(branches), batch_size), desc="Processing branches", leave=False):
    #     batch: list[str] = branches[i:i+batch_size]
    #     monthly_returns_df: pd.DataFrame = batch_processor(batch)
    #     monthly_returns_df.to_parquet(f'./monthly_returns/monthly_returns_{i//batch_size}.parquet')
        
    monthly_returns_df: pd.DataFrame = gather_monthly_returns('./monthly_returns')
    monthly_returns_df.index = pd.to_datetime(monthly_returns_df.index)
    
    # replace all nans and infs with 1
    monthly_returns_df.replace([np.inf, -np.inf], 1, inplace=True)
    back_start: str = "2010-02"
    back_end: str = "2018-12"
    forward_start: str = "2019-01"
    forward_end: str = "2024-06"
    top_n: int = 5000

    dataset: pd.DataFrame = one_back_one_forward(monthly_returns_df, back_start, back_end, forward_start, forward_end, top_n)

    # save the dataset to a csv file
    dataset.to_csv(f'./dataset_{back_start}_{back_end}_{forward_start}_{forward_end}_{top_n}_2.csv')