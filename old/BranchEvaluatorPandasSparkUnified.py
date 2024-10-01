import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, year, month, when
from tqdm import tqdm

def create_spark_session():
    return SparkSession.builder \
        .appName("ParquetWriter") \
        .getOrCreate()

def load_data(spark, ticker):
    file_path = f'./indicator_data/{ticker}.csv'
    df = spark.read.csv(file_path, header=True, inferSchema=True, timestampFormat="yyyy-MM-dd")
    df = df.withColumnRenamed('date', 'date')
    return df

def evaluate_condition(df, indicator, direction, value, days):
    if direction == '>':
        condition = df[indicator] > value
    else:
        condition = df[indicator] < value

    # Spark doesn't have a direct rolling function like pandas, so we implement a window-based approach
    from pyspark.sql.window import Window
    from pyspark.sql.functions import min as spark_min

    window_spec = Window.orderBy("date").rowsBetween(-days + 1, 0)
    rolling_condition = spark_min(when(condition, 1).otherwise(0)).over(window_spec)

    return rolling_condition

def evaluate_branch(branch, spark):
    parts = branch.split('-')
    indicator, period, indicator_ticker, direction, threshold, days, trading_ticker = parts

    # Load data for indicator and trading
    indicator_df = load_data(spark, indicator_ticker)
    trading_df = load_data(spark, trading_ticker)

    # Inner join on date
    common_df = indicator_df.join(trading_df, on="date", how="inner")

    # Evaluate the condition
    condition_met = evaluate_condition(common_df, f"{indicator}_{period}", direction, float(threshold), int(days))

    # Create new columns for the condition and returns
    common_df = common_df.withColumn("condition_met", condition_met)
    common_df = common_df.withColumn("shifted_condition", col("condition_met").shift(1))
    common_df = common_df.withColumn("trade_returns_day", when(col("shifted_condition") == 1, col("close").pct_change() + 1).otherwise(1))

    return common_df

def save_as_parquet(df, branch, output_dir):
    # Add year, month, and branch columns for partitioning
    df = df.withColumn('year', year(col('date')))
    df = df.withColumn('month', month(col('date')))
    df = df.withColumn('branch', lit(branch))

    # Write partitioned Parquet data
    df.write.partitionBy("year", "month", "branch").mode("append").parquet(output_dir)

def process_branch(branch, spark, output_dir):
    try:
        result_df = evaluate_branch(branch, spark)
        save_as_parquet(result_df, branch, output_dir)
    except Exception as e:
        return f"Error processing branch {branch}: {str(e)}"

def main():
    # Create a Spark session
    spark = create_spark_session()

    with open('branches.txt', 'r') as f:
        branches = f.read().splitlines()

    output_dir = './output_data_spark'
    os.makedirs(output_dir, exist_ok=True)

    # Process branches in parallel using Spark's distributed processing
    for branch in tqdm(branches, desc="Processing branches", unit="branch"):
        process_branch(branch, spark, output_dir)

    print("Branch evaluation completed.")

if __name__ == "__main__":
    main()
