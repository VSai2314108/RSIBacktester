from deephaven_server import Server

server = Server(
    port=10001,
    jvm_args=[
        "-Xmx12g",
        "-DAuthHandlers=io.deephaven.auth.AnonymousAuthenticationHandler",
    ],
)

server.start()

from deephaven import parquet
from deephaven.table import Table
from deephaven import merge
from deephaven.parquet import ParquetFileLayout
import os
from datetime import timedelta
import pandas as pd
from deephaven import numpy as np
from tqdm import tqdm
from deephaven import agg
from deephaven import new_table

def read_parquet_files(input_path: str, start_date: str, end_date: str) -> Table:
    # Define the table definition using InputColumn
    # table_def = [
    #     # Column("year", dht.string, column_type=ColumnType.PARTITIONING),
    #     # Column("month", dht.string, column_type=ColumnType.PARTITIONING),
    #     Column("branch", dht.string, column_type=ColumnType.PARTITIONING),
    #     Column("date", dht.long),
    #     Column("condition_met", dht.int32),
    #     Column("trade_returns_day", dht.double)
    # ]

    # Create an empty table with the defined schema
    tables = []

    start_year, start_month = start_date[:4], start_date[5:7]
    end_year, end_month = end_date[:4], end_date[5:7]
    current_year, current_month = int(start_year), int(start_month)
    
    # create a list of month year tuples (int, int)
    months = []
    while (current_year, current_month) <= (int(end_year), int(end_month)):
        months.append((current_year, current_month))
        # Move to the next month
        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
        
    for year, month in months:
        path = f"{input_path}/year={year}/month={month}"
        if os.path.exists(path):
            table = parquet.read(path, file_layout=ParquetFileLayout.KV_PARTITIONED)
            if table is not None and table.size > 0:
                # Use merge to combine the tables
                tables.append(table)
        else: 
            return None # we want to evaluate only if the full date range is present


    combined_table = None
    if len(tables) != 0:
        combined_table: Table = merge(tables)
        
        # replace all null values in the trade_returns_day column with 1
        combined_table = combined_table.update("trade_returns_day = (trade_returns_day == NULL_DOUBLE) ? 1.0 : trade_returns_day")
    
    return combined_table



# count works need to make sure all methods have a vector method in java and to use simpler
def calculate_metrics(df: Table):
    results_table: Table = df.agg_by(
        aggs=[
            agg.formula('(product(x) - 1) * 100', 'x', ['cum_return = trade_returns_day']),
            agg.formula('((product(x) - 1) / count(x)) * 252 * 100', 'x', ['cagr = trade_returns_day']),
            agg.formula('min(cumprod(x)) * 100', 'x', ['max_drawdown = trade_returns_day']),
            # agg.formula('sum(x != 1) / count(x) * 100', ['x'], ['percent_days_in_market = trade_returns_day']),
            agg.formula('((product(x) - 1) * 100) / min(cumprod(x))', 'x', ['romad = trade_returns_day']),
            # agg.sorted_last('x', 'x', ['date = date'])
        ],
        by=['branch']
    )
    
    print(results_table.head(20).to_string())
    
    # # convert this to seperate update statements
    # df = df.update('cum_return = (exp(sum(log(trade_returns_day))) - 1) * 100')
    # df = df.update('cagr = (pow(1 + (exp(sum(log(trade_returns_day))) - 1), 1 / (count(trade_returns_day) / 252.0)) - 1) * 100')
    # df = df.update('max_drawdown = (1 - min(cumprod(trade_returns_day))) * 100')
    # df = df.update('in_market = (trade_returns_day != 1) ? 1 : 0')
    # df = df.update('percent_days_in_market = sum(in_market) / count(in_market) * 100')
    # df = df.update('romad = cagr / max_drawdown * sqrt(252 / count(trade_returns_day))')
    
    # drop the year and month columns
    # df = df.drop_columns(["in_market"])
    # results = results_table.sort("date").last_by("branch")
    
        
    return results_table

def select_top_branches(df, branch_count: int, min_return: float):
    return df.where(f"cagr >= {min_return}").sort_descending("romad").head(branch_count)

def process_date_range(current_date, look_back_window, branch_count, min_return, look_forward_window, input_path):
    back_start_date = (current_date - timedelta(days=look_back_window))
    back_end_date = current_date - timedelta(days=1)
    forward_end_date = current_date + timedelta(days=look_forward_window)
    
    # print(f"Back start date: {back_start_date.strftime('%Y-%m-%d')}")
    # print(f"Back end date: {back_end_date.strftime('%Y-%m-%d')}")
    # print(f"Forward end date: {forward_end_date.strftime('%Y-%m-%d')}")
    
    df = read_parquet_files(input_path, back_start_date.strftime("%Y-%m-%d"), forward_end_date.strftime("%Y-%m-%d"))
    if df is None or df.size == 0:
        print(f"No data found for {current_date.strftime('%Y-%m')}")
        return None
    
    # print(f"Read {df.size} rows for {current_date.strftime('%Y-%m')}")
    
    back_df = df.where(f"date <= '{back_end_date.strftime('%Y-%m-%d')}'")
    forward_df = df.where(f"date > '{back_end_date.strftime('%Y-%m-%d')}'")
    
    # print(f"Back df size: {back_df.size}")
    # print(f"Forward df size: {forward_df.size}")

    
    # print(back_df.head(20).to_string())
    
    back_metrics = calculate_metrics(back_df)
    top_branches = select_top_branches(back_metrics, branch_count, min_return)
    
    # print(top_branches.head(20).to_string())
    
    forward_metrics = calculate_metrics(forward_df)
    
    combined_metrics = top_branches.join(forward_metrics, on='branch', joins=[f'forward_{col.name} = {col.name}' for col in forward_metrics.columns if col.name != 'branch'])
    print(combined_metrics.head(200).to_string())
    
    # Create a Deephaven table for period results
    aggregated_metrics = combined_metrics.agg_by(
        aggs=[
            agg.formula('avg(forward_cagr)', 'x', ['forward_cagr = forward_cagr']),
            agg.formula('avg(cagr)', 'x', ['cagr = cagr'])
        ],
    )
    print(aggregated_metrics.head(5).to_string())
    
    parquet.write(aggregated_metrics, f"results-deephaven/metrics_{forward_end_date.strftime('%Y_%m')}.parquet")
    
    return aggregated_metrics

def main():
    # Parameters
    look_back_window = 1080
    branch_count = 5000
    min_return = 5
    look_forward_window = 1080
    input_path = "output_data_spark"
    
    start_date = pd.to_datetime("2006-02-01")
    end_date = pd.to_datetime("2023-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    results = []
    for date in tqdm(date_range, desc="Processing dates"):
        result = process_date_range(date, look_back_window, branch_count, min_return, look_forward_window, input_path)
        if result is not None:
            results.append(result)
    
    # Merge all period results into a single table
    if results:
        final_results = merge(results)
        parquet.write(final_results, "results-deephaven/monthly_returns.parquet")
    else:
        print("No results to process.")

if __name__ == "__main__":
    if not os.path.exists("results-deephaven"):
        os.makedirs("results-deephaven")
    main()
