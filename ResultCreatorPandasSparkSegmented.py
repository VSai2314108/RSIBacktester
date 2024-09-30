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
from deephaven import agg
import datetime
from typing import List
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_branch(branch, year, month, input_path):
    path = os.path.join(input_path, f"year={year}", f"month={month}", f"branch={branch}")
    if os.path.exists(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]
        branch_tables = []
        for file in files:
            table = parquet.read(file)
            if table is not None and table.size > 0:
                table = table.lazy_update(formulas=[f"branch=`{branch}`"])
            branch_tables.append(table)
        return branch_tables
    return []

def get_months_between(start_date, end_date):
    months = []
    current_year = start_date.year
    current_month = start_date.month
    while (current_year, current_month) <= (end_date.year, end_date.month):
        months.append((current_year, current_month))
        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
    return months

def read_branches_data(branches: List[str], start_date, end_date, input_path: str) -> Table:
    months = get_months_between(start_date, end_date)
    tables = []
    for year, month in months:
        # print(f"Year: {year}, Month: {month}")
        # with Pool(processes=cpu_count()-1) as pool:
        #     args_list = [(branch, year, month, input_path) for branch in branches]
        #     results = list(tqdm(pool.starmap(process_branch, args_list), total=len(args_list), desc="Processing branches", leave=False))
            
        # tables.extend([table for branch_tables in results for table in branch_tables])
        
        for branch in tqdm(branches, desc="Processing branches", leave=False):
            path = os.path.join(input_path, f"year={year}", f"month={month}", f"branch={branch}")
            if os.path.exists(path):
                files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]
                branch_tables = []
                for file in files:
                    table = parquet.read(file)
                    if table is not None and table.size > 0:
                        table = table.lazy_update(formulas=[f"branch=`{branch}`"])
                tables.append(table)
    if tables:
        combined_table = merge(tables)
        # Filter data by date range
        combined_table = combined_table.where(filters = 
                                              [f"date >= '{start_date.strftime('%Y-%m-%d')}'",
                                               f"date <= '{end_date.strftime('%Y-%m-%d')}'"]
                                              )
        return combined_table
    else:
        return None

# def read_branches_data(branches: List[str], start_date: datetime.date, end_date: datetime.date, input_path: str) -> Table:
#     months = get_months_between(start_date, end_date)
    
#     start_year = start_date.year
#     end_year = end_date.year
#     start_month = start_date.month
#     end_month = end_date.month
    
    
#     combined_table = parquet.read(input_path).where(
#         filters=[
#             f"year >= {start_year}",
#             f"year <= {end_year}",
#             f"month >= {start_month}",
#             f"month <= {end_month}",
#             f"branch in {','.join([f'`{branch}`' for branch in branches])}"
#         ]
#     )
    
#     print("Finished reading data")
    
#     if combined_table:
#         combined_table = combined_table.where(f"date >= '{start_date.strftime('%Y-%m-%d')}' and date <= '{end_date.strftime('%Y-%m-%d')}'")
#         return combined_table
#     else:
#         return None
    
def calculate_metrics(df: Table):
    print("Calculating metrics")
    results_table: Table = df.agg_by(
        aggs=[
            agg.formula('(product(x) - 1) * 100', 'x', ['cum_return = trade_returns_day']),
            agg.formula('((product(x) - 1) / count(x)) * 252 * 100', 'x', ['cagr = trade_returns_day']),
            agg.formula('min(cumprod(x)) * 100', 'x', ['max_drawdown = trade_returns_day']),
            agg.formula('((product(x) - 1) * 100) / min(cumprod(x))', 'x', ['romad = trade_returns_day']),
        ],
        by=['branch']
    )
    return results_table

def select_top_branches(df, branch_count: int, min_return: float):
    return df.where(f"cagr >= {min_return}").sort_descending("romad").head(branch_count)

def process_date_range(current_date, look_back_window, branch_count, min_return, look_forward_window, input_path, batch_size):
    back_start_date = current_date - timedelta(days=look_back_window)
    back_end_date = current_date - timedelta(days=1)
    forward_end_date = current_date + timedelta(days=look_forward_window)
    
    back_months = get_months_between(back_start_date, back_end_date)
    back_branches = set()

    # Collect all branches in the back period
    for year, month in back_months:
        path = f"{input_path}/year={year}/month={month}"
        if os.path.exists(path):
            # List branch directories
            branches = [d.replace('branch=', '') for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            back_branches.update(branches)
    
    back_branches = list(back_branches)
    
    if not back_branches:
        print(f"No branches found for back period ending {back_end_date.strftime('%Y-%m')}")
        return None
    
    branch_metrics_list = []
    
    # Process branches in batches
    for i in range(0, len(back_branches), batch_size):
        batch_branches = back_branches[i:i+batch_size]
        print(f"Processing back period batch {i//batch_size + 1}: branches {i+1} to {i+len(batch_branches)}")
        back_data = read_branches_data(batch_branches, back_start_date, back_end_date, input_path)
        print(f"Back data: {back_data.head(10).to_string()}")
        if back_data is not None and back_data.size > 0:
            back_metrics = calculate_metrics(back_data)
            branch_metrics_list.append(back_metrics)
    
    if not branch_metrics_list:
        print(f"No data found for back period ending {back_end_date.strftime('%Y-%m')}")
        return None
    
    # Merge metrics for all batches
    branch_metrics_table = merge(branch_metrics_list)
    
    # Select top branches
    top_branches = select_top_branches(branch_metrics_table, branch_count, min_return)
    top_branches_list = top_branches.to_pandas()['branch'].tolist()
    
    forward_metrics_list = []
    
    # Process top branches in batches
    for i in range(0, len(top_branches_list), batch_size):
        batch_branches = top_branches_list[i:i+batch_size]
        print(f"Processing forward period batch {i//batch_size + 1}: branches {i+1} to {i+len(batch_branches)}")
        forward_data = read_branches_data(batch_branches, back_end_date + timedelta(days=1), forward_end_date, input_path)
        if forward_data is not None and forward_data.size > 0:
            forward_metrics = calculate_metrics(forward_data)
            forward_metrics_list.append(forward_metrics)
    
    if not forward_metrics_list:
        print(f"No forward data found for date {current_date.strftime('%Y-%m')}")
        return None
    
    forward_metrics_table = merge(forward_metrics_list)
    
    # Combine back and forward metrics
    combined_metrics = top_branches.join(
        forward_metrics_table, 
        on='branch', 
        joins=[f'forward_{col.name} = {col.name}' for col in forward_metrics_table.columns if col.name != 'branch']
    )
    
    # Aggregate metrics
    aggregated_metrics = combined_metrics.agg_by(
        aggs=[
            agg.avg('forward_cagr'),
            agg.avg('cagr'),
        ]
    )
    
    # Save results
    parquet.write(aggregated_metrics, f"results-deephaven/metrics_{forward_end_date.strftime('%Y_%m')}.parquet")
    
    return aggregated_metrics

def main():
    # Parameters
    look_back_window = 1080
    branch_count = 5000
    min_return = 5
    look_forward_window = 1080
    input_path = "output_data_spark"
    batch_size = 1000  # Set your desired batch size here
    
    start_date = pd.to_datetime("2006-02-01")
    end_date = pd.to_datetime("2023-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    results = []
    for date in tqdm(date_range, desc="Processing dates"):
        result = process_date_range(date, look_back_window, branch_count, min_return, look_forward_window, input_path, batch_size)
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
