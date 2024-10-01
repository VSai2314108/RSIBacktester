from deephaven_server import Server
server = Server(
    port=10000,
    jvm_args=[
        "-Xmx4g",
        "-DAuthHandlers=io.deephaven.auth.AnonymousAuthenticationHandler",
    ],
)
server.start()


from datetime import datetime
import os
from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from deephaven import pandas as dhpd
from deephaven import parquet
from deephaven.table import Table
from deephaven import new_table
from deephaven import agg, merge
import deephaven.dtypes as dht
from deephaven import merge
import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


from deephaven import parquet
from deephaven.table import Table
import os
import logging
from datetime import datetime
from deephaven import dtypes as dht
from deephaven import time as dhtime

def read_parquet_files(input_path: str, start_date: str, end_date: str) -> Table:
    start_year, start_month = start_date[:4], start_date[5:7]
    end_year, end_month = end_date[:4], end_date[5:7]
    
    combined_table = None
    for year in range(int(start_year), int(end_year) + 1):
        start_month_in_year = start_month if year == int(start_year) else '01'
        end_month_in_year = end_month if year == int(end_year) else '12'
        
        for month in range(int(start_month_in_year), int(end_month_in_year) + 1):
            month_str = f"{month:02d}"
            path = f"{input_path}/year={year}/month={month_str}"
            logger.info(f"Checking path: {path}")
            if os.path.exists(path):
                try:
                    table = parquet.read(path)
                    if table is None or table.size == 0:
                        logger.warning(f"Empty or None table read from {path}")
                        continue
                    if combined_table is None:
                        combined_table = table
                    else:
                        combined_table = combined_table.union(table)
                    logger.info(f"Successfully read and combined data from {path}")
                except Exception as e:
                    logger.error(f"Error reading or combining data from {path}: {str(e)}")
            else:
                logger.warning(f"Path does not exist: {path}")
    
    if combined_table is None:
        logger.warning(f"No data found for date range {start_date} to {end_date}")
        return None
    
    print(combined_table.head(5).to_string())
    
    return combined_table


def calculate_metrics(df):
    return (df.by(
        'branch',
        cum_return = agg.custom("(Math.exp(sum(Math.log(trade_returns_day))) - 1) * 100"),
        days = agg.count_("date"),
        years = agg.custom("count(date) / 252.0"),
        cagr = agg.custom("(Math.pow(1 + (Math.exp(sum(Math.log(trade_returns_day))) - 1), 1 / (count(date) / 252.0)) - 1) * 100"),
        max_drawdown = agg.custom("Math.abs(min((cumprod(1 + trade_returns_day) - cummax(cumprod(1 + trade_returns_day))) / cummax(cumprod(1 + trade_returns_day)))) * 100"),
        percent_days_in_market = agg.custom("sum(trade_returns_day != 1) / count(date) * 100")
    ).update("romad = cagr / max_drawdown * Math.sqrt(252 / days)"))

def select_top_branches(df, branch_count: int, min_return: float):
    return df.where(f"cagr >= {min_return}").sort_descending("romad").head(branch_count)

def process_date_range(current_date, look_back_window, branch_count, min_return, look_forward_window, input_path):
    back_start_date = (current_date - timedelta(days=look_back_window)).replace(day=1)
    back_end_date = current_date - timedelta(days=1)
    forward_end_date = current_date + timedelta(days=look_forward_window)
    
    df = read_parquet_files(input_path, back_start_date.strftime("%Y-%m-%d"), forward_end_date.strftime("%Y-%m-%d"))
    if df is None:
        print(f"No data found for {current_date.strftime('%Y-%m')}")
        return None
    
    print(f"Read {df.size} rows for {current_date.strftime('%Y-%m')}")
    
    back_df = df.where(f"date <= '{back_end_date.strftime('%Y-%m-%d')}'")
    forward_df = df.where(f"date > '{back_end_date.strftime('%Y-%m-%d')}'")
    
    back_metrics = calculate_metrics(back_df)
    top_branches = select_top_branches(back_metrics, branch_count, min_return)
    
    forward_metrics = calculate_metrics(forward_df)
    
    combined_metrics = merge.inner_join(top_branches, forward_metrics, on=['branch'], 
                                        lsuffix='_back', rsuffix='_forward')
    
    period_results = {
        "period": current_date.strftime("%Y-%m"),
        "avg_back_cagr": combined_metrics.avg("cagr_back"),
        "avg_forward_cagr": combined_metrics.avg("cagr_forward")
    }
    
    parquet.write(combined_metrics, f"results-deephaven/metrics_{current_date.strftime('%Y_%m')}.parquet")
    
    return period_results

def main():
    # Parameters
    look_back_window = 90
    branch_count = 100
    min_return = 10
    look_forward_window = 30
    input_path = "output_data_spark"
    
    start_date = pd.to_datetime("2000-01-01")
    end_date = pd.to_datetime("2023-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    results = []
    for date in tqdm(date_range, desc="Processing dates"):
        result = process_date_range(date, look_back_window, branch_count, min_return, look_forward_window, input_path)
        if result:
            results.append(result)
    
    results_df = pd.DataFrame(results)
    results_table = new_table(results_df)
    dhpd.write_csv(results_table, "results-deephaven/monthly_returns.csv")

if __name__ == "__main__":
    if not os.path.exists("results-deephaven"):
        os.makedirs("results-deephaven")
    main()
