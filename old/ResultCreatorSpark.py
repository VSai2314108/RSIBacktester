import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, year, month, when, avg, min as spark_min, max as spark_max, sum as spark_sum, log, exp
from pyspark.sql.window import Window
from datetime import timedelta
import pandas as pd
from tqdm import tqdm

def create_spark_session():
    return SparkSession.builder \
        .appName("BranchProcessing") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

def process_branch(spark, branch, year, month, input_path):
    path = os.path.join(input_path, f"year={year}", f"month={month}", f"branch={branch}")
    if os.path.exists(path):
        df = spark.read.parquet(path)
        if df is not None and df.count() > 0:
            df = df.withColumn("branch", lit(branch))
            return df
    return None

def get_months_between(start_date, end_date):
    months = []
    current_year = start_date.year
    current_month = start_date.month
    while (current_year, current_month) <= (end_date.year, end_date.month):
        months.append((current_year, current_month))
        if current_month == 12:
            current_month = 1
            current_year +=1
        else:
            current_month +=1
    return months

def read_branches_data(spark, branches, start_date, end_date, input_path):
    months = get_months_between(start_date, end_date)
    dataframes = []
    for year, month in tqdm(months, desc="Processing months", unit="month"):
        for branch in branches:
            df = process_branch(spark, branch, year, month, input_path)
            if df is not None:
                dataframes.append(df)
    if dataframes:
        combined_df = dataframes[0]
        for df in dataframes[1:]:
            combined_df = combined_df.union(df)
        # Filter data by date range
        combined_df = combined_df.filter((col("date") >= lit(start_date.strftime('%Y-%m-%d'))) & (col("date") <= lit(end_date.strftime('%Y-%m-%d'))))
        return combined_df
    else:
        return None

def calculate_metrics(df):
    # Sort the data by date per branch
    window_spec = Window.partitionBy('branch').orderBy('date')

    # Compute daily log returns
    df = df.withColumn('log_return', log(col('trade_returns_day')))

    # Compute cumulative log returns per branch
    df = df.withColumn('cum_log_return', spark_sum('log_return').over(window_spec))

    # Exponentiate cumulative log returns to get cumulative returns
    df = df.withColumn('cum_return_daily', exp(col('cum_log_return')))

    # Compute rolling maximum of cumulative returns to compute drawdowns
    df = df.withColumn('rolling_max', spark_max('cum_return_daily').over(window_spec))

    # Compute drawdown
    df = df.withColumn('drawdown', col('cum_return_daily') / col('rolling_max') - 1)

    # Compute max drawdown per branch
    max_drawdown_df = df.groupBy('branch').agg(
        spark_min('drawdown').alias('max_drawdown')
    )

    # Compute total cumulative return per branch
    total_return_df = df.groupBy('branch').agg(
        (exp(spark_sum('log_return')) - 1).alias('cum_return')
    )

    # Compute ROMAD
    metrics_df = total_return_df.join(max_drawdown_df, on='branch', how='inner')
    metrics_df = metrics_df.withColumn('romad', col('cum_return') / abs(col('max_drawdown')))

    # Compute CAGR
    date_diff_df = df.groupBy('branch').agg(
        (max('date').cast('long') - min('date').cast('long')).alias('days')
    )
    metrics_df = metrics_df.join(date_diff_df, on='branch', how='inner')
    metrics_df = metrics_df.withColumn('years', col('days') / (365.25 * 24 * 3600))
    metrics_df = metrics_df.withColumn('cagr', (col('cum_return') + 1).pow(1 / col('years')) - 1)

    return metrics_df.select('branch', 'cum_return', 'cagr', 'max_drawdown', 'romad')

def select_top_branches(df, branch_count, min_return):
    df_filtered = df.filter(df['cagr'] >= min_return)
    df_sorted = df_filtered.orderBy(df_filtered['romad'].desc())
    top_branches = df_sorted.limit(branch_count)
    return top_branches

def process_date_range(spark, current_date, look_back_window, branch_count, min_return, look_forward_window, input_path, batch_size):
    back_start_date = current_date - timedelta(days=look_back_window)
    back_end_date = current_date - timedelta(days=1)
    forward_end_date = current_date + timedelta(days=look_forward_window)

    back_months = get_months_between(back_start_date, back_end_date)
    back_branches = set()

    # Collect all branches in the back period
    last_year, last_month = back_months[-1]
    path = f"{input_path}/year={last_year}/month={last_month}"
    if os.path.exists(path):
        # List branch directories
        branches = [d.replace('branch=', '') for d in os.listdir(path) if d.startswith('branch=') and os.path.isdir(os.path.join(path, d))]
        back_branches.update(branches)

    back_branches = list(back_branches)

    if not back_branches:
        print(f"No branches found for back period ending {back_end_date.strftime('%Y-%m')}")
        return None

    branch_metrics_list = []

    # Process branches in batches
    for i in tqdm(range(0, len(back_branches), batch_size), desc="Processing back period batches"):
        batch_branches = back_branches[i:i+batch_size]
        print(f"Processing back period batch {i//batch_size + 1}: branches {i+1} to {i+len(batch_branches)}")
        back_data = read_branches_data(spark, batch_branches, back_start_date, back_end_date, input_path)
        if back_data is not None and back_data.count() > 0:
            back_metrics = calculate_metrics(back_data)
            # Write intermediate metrics to Parquet to avoid keeping data in memory
            temp_output = f"temp_metrics/back_metrics_{i//batch_size + 1}.parquet"
            back_metrics.write.mode('overwrite').parquet(temp_output)
            branch_metrics_list.append(temp_output)

    if not branch_metrics_list:
        print(f"No data found for back period ending {back_end_date.strftime('%Y-%m')}")
        return None

    # Read all intermediate metrics
    branch_metrics_df = spark.read.parquet(*branch_metrics_list)

    # Select top branches
    top_branches_df = select_top_branches(branch_metrics_df, branch_count, min_return)
    top_branches_list = [row['branch'] for row in top_branches_df.collect()]

    forward_metrics_list = []

    # Process top branches in batches
    for i in range(0, len(top_branches_list), batch_size):
        batch_branches = top_branches_list[i:i+batch_size]
        print(f"Processing forward period batch {i//batch_size + 1}: branches {i+1} to {i+len(batch_branches)}")
        forward_data = read_branches_data(spark, batch_branches, back_end_date + timedelta(days=1), forward_end_date, input_path)
        if forward_data is not None and forward_data.count() > 0:
            forward_metrics = calculate_metrics(forward_data)
            # Write intermediate metrics to Parquet to avoid keeping data in memory
            temp_output = f"temp_metrics/forward_metrics_{i//batch_size + 1}.parquet"
            forward_metrics.write.mode('overwrite').parquet(temp_output)
            forward_metrics_list.append(temp_output)

    if not forward_metrics_list:
        print(f"No forward data found for date {current_date.strftime('%Y-%m')}")
        return None

    # Read all intermediate forward metrics
    forward_metrics_df = spark.read.parquet(*forward_metrics_list)

    # Combine back and forward metrics
    combined_metrics = top_branches_df.join(
        forward_metrics_df.select('branch', 
                                  col('cum_return').alias('forward_cum_return'), 
                                  col('cagr').alias('forward_cagr')),
        on='branch',
        how='inner'
    )

    # Aggregate metrics
    aggregated_metrics = combined_metrics.agg(
        avg('forward_cagr').alias('avg_forward_cagr'),
        avg('cagr').alias('avg_back_cagr')
    )

    # Save results
    output_path = f"results-spark/metrics_{forward_end_date.strftime('%Y_%m')}.parquet"
    aggregated_metrics.write.mode('overwrite').parquet(output_path)

    # Clean up temporary files
    for temp_file in branch_metrics_list + forward_metrics_list:
        os.remove(temp_file)

    return aggregated_metrics

def main():
    # Create a Spark session
    spark = create_spark_session()

    # Parameters
    look_back_window = 1080
    branch_count = 5000
    min_return = 5
    look_forward_window = 1080
    input_path = "output_data_spark"
    batch_size = 100  # Adjust based on available memory

    start_date = pd.to_datetime("2006-02-01")
    end_date = pd.to_datetime("2023-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # For testing purposes, use a single date
    single_date = pd.to_datetime("2019-01-01")
    date_range = pd.date_range(start=single_date, periods=1, freq='MS')

    look_back_window = 3650
    look_forward_window = 3650

    if not os.path.exists("temp_metrics"):
        os.makedirs("temp_metrics")

    results = []
    for date in date_range:
        result = process_date_range(spark, date, look_back_window, branch_count, min_return, look_forward_window, input_path, batch_size)
        if result is not None:
            results.append(result)

    # Merge all period results into a single DataFrame
    if results:
        final_results_df = results[0]
        for df in results[1:]:
            final_results_df = final_results_df.union(df)
        final_output_path = "results-spark/monthly_returns.parquet"
        final_results_df.write.mode('overwrite').parquet(final_output_path)
    else:
        print("No results to process.")

    # Clean up temporary directory
    os.rmdir("temp_metrics")

if __name__ == "__main__":
    if not os.path.exists("results-spark"):
        os.makedirs("results-spark")
    main()
