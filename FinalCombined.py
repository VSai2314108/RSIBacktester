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
    def clean_series(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan).dropna()

    def cagr(series: pd.Series) -> float:
        series = clean_series(series)
        if len(series) == 0:
            return 0
        percent_return = (series.prod() - 1) # as decimal
        cagr = (((1+percent_return) ** (365 / len(series))) - 1) * 100 # as percent
        return cagr
    
    def days_in_market_ratio(series: pd.Series) -> float:
        if len(series) == 0:
            return 0
        return (series != 1).sum() / len(series)
    
    def calmar(series: pd.Series) -> float:
        series = clean_series(series)
        if len(series) == 0:
            return 0
        
        percent_return = (series.prod() - 1) # as decimal
        cagr_val = (((1+percent_return) ** (365 / len(series))) - 1) * 100
        max_drawdown = max([1 - min(series.cumprod()), 0.000001])        
        return cagr_val / max_drawdown if max_drawdown != 0 else 0
    
    def gross_return(series: pd.Series) -> float:
        series = clean_series(series)
        return series.prod()
    
    def tuple_data(series: pd.Series) -> tuple[float, float, float]:
        return (cagr(series), days_in_market_ratio(series), calmar(series), gross_return(series))
    
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    # Clean the series once
    # df['trade_returns_day'] = clean_series(df['trade_returns_day'])
        
    monthly_returns = df.groupby(['year', 'month'])['trade_returns_day'].agg(tuple_data)
    
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
    dataframes = []
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')][:100]
    for file in tqdm(files, desc="Processing files", unit="file"):
        df = pd.read_parquet(os.path.join(directory, file))
        dataframes.append(df)
    monthly_returns_df = pd.concat(dataframes, axis=1, join='inner')  # Use join='inner' to keep only common indices
    
    # print the data for this column specifically - rsi-3-SPXL->-23-1-ERY
    print(monthly_returns_df['rsi-3-SPXL->-23-1-ERY'])
    
    # write it to a temp file
    monthly_returns_df['rsi-3-SPXL->-23-1-ERY'].to_csv(f'./temp/tmp.csv')
    
    return monthly_returns_df


def one_back_one_forward(monthly_returns_df: pd.DataFrame, 
                         back_start: Union[str, pd.Timestamp], 
                         back_end: Union[str, pd.Timestamp], 
                         forward_start: Union[str, pd.Timestamp], 
                         forward_end: Union[str, pd.Timestamp], 
                         top_n: int,
                         min_cagr: float,
                         min_days_in_market: float) -> pd.DataFrame:
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
    
    # Calculate average metrics for the back period
    back_period_metrics = back_returns.apply(lambda col: pd.Series({
        'cagr': np.mean([x[0] for x in col]),
        'days_in_market': np.mean([x[1] for x in col]),
        'calmar': np.mean([x[2] for x in col]),
        'gross_return_percent': (np.prod([x[3] for x in col if x[3] != 0]) - 1) * 100
    }))
    
    # make the columns the keys and the values the series by transposing the dataframe
    back_period_metrics = back_period_metrics.T
    
    # print(back_period_metrics.head())
    
    # Filter branches based on minimum CAGR and days in market ratio
    filtered_branches = back_period_metrics[
        (back_period_metrics['cagr'] >= min_cagr) & 
        (back_period_metrics['days_in_market'] >= min_days_in_market)
    ]
    
    branches_to_check: list[str] = filtered_branches.nlargest(top_n, 'calmar').index.tolist()
    # print(branches_to_check)
    
    top_n_branches = back_period_metrics.loc[branches_to_check]
    # print(top_n_branches.head())
    
    # Calculate average metrics for the forward period
    forward_period_metrics = forward_returns[branches_to_check].apply(lambda col: pd.Series({
        'cagr': np.mean([x[0] for x in col]),
        'days_in_market': np.mean([x[1] for x in col]),
        'calmar': np.mean([x[2] for x in col]),
        'gross_return_percent': (np.prod([x[3] for x in col if x[3] != 0]) - 1) * 100
    }))
    
    forward_period_metrics = forward_period_metrics.T
    
    # print(forward_period_metrics.head())
    
    # prefix both dataframes column names
    top_n_branches.columns = [f'back_{col}' for col in top_n_branches.columns]
    forward_period_metrics.columns = [f'forward_{col}' for col in forward_period_metrics.columns]
    
    # Combine the back period and forward period metrics
    dataset: pd.DataFrame = pd.concat([top_n_branches, forward_period_metrics], axis=1)
    # print(dataset.head())
    # dataset.columns = ['back_cagr', 'back_days_in_market', 'back_calmar', 
    #                    'forward_cagr', 'forward_days_in_market', 'forward_calmar']
    
    # Add a row for the averages
    dataset.loc['avg'] = dataset.mean()
    
    # Calculate the portion of year for back and forward periods
    # dataset.loc['portion_of_year'] = [(back_end - back_start).days / 365, (forward_end - forward_start).days / 365]
    
    # print(dataset)
    
    return dataset

if __name__ == "__main__":
    # os.makedirs('./monthly_returns_3', exist_ok=True)
    
    branches: list[str] = []
    with open('branches.txt', 'r') as file:
        branches = [line.strip() for line in file]
        
    batch_size: int = 100
    for i in tqdm(range(0, len(branches), batch_size), desc="Processing branches", leave=False):
        batch: list[str] = branches[i:i+batch_size]
        monthly_returns_df: pd.DataFrame = batch_processor(batch)
        monthly_returns_df.to_parquet(f'./monthly_returns_3/monthly_returns_{i//batch_size}.parquet')
        
    monthly_returns_df: pd.DataFrame = gather_monthly_returns('./monthly_returns_3')
    monthly_returns_df.index = pd.to_datetime(monthly_returns_df.index)
    
    # print(monthly_returns_df.head())
    
    
    # replace all nans and infs with appropriate values
    def clean_func(data) -> list[float]:
        # replace any nan or inf values appropriately
        if data is None:
            return [0, 0, 0, 1]
        
        if type(data) != np.ndarray: # eliminate NaN
            return [0, 0, 0, 1]
        
        new_data = [0, 0, 0, 1]
        
        for i, elem in enumerate(data):
            # if the is not nan or inf replace it with the value
            if np.isfinite(elem):
                new_data[i] = elem
        return new_data
    monthly_returns_df = monthly_returns_df.applymap(clean_func)
    
    back_start: str = "2010-02"
    back_end: str = "2015-12"
    forward_start: str = "2015-01"
    forward_end: str = "2018-06"
    top_n: int = 5000
    min_cagr: float = 5  # 5% minimum CAGR
    min_days_in_market: float = 0.3  # 30% minimum days in market ratio

    dataset: pd.DataFrame = one_back_one_forward(monthly_returns_df, back_start, back_end, forward_start, forward_end, top_n, min_cagr, min_days_in_market)

    # save the dataset to a csv file
    dataset.to_csv(f'./dataset_{back_start}_{back_end}_{forward_start}_{forward_end}_{top_n}_min_cagr_{min_cagr}_min_dim_{min_days_in_market}.csv')