import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, year, month, when, lag, min as spark_min
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType
from tqdm import tqdm
from typing import List

def create_spark_session() -> SparkSession:
    print("Creating Spark session...")
    return SparkSession.builder \
        .appName("ParquetWriter") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

def load_data(spark: SparkSession, ticker: str) -> DataFrame:
    file_path = f'./indicator_data/{ticker}.csv'
    print(f"Loading data from {file_path}...")
    df = spark.read.csv(file_path, header=True, inferSchema=True, timestampFormat="yyyy-MM-dd")
    return df

def evaluate_condition(df: DataFrame, indicator: str, direction: str, value: float, days: int) -> Column:
    print(f"Evaluating condition: {indicator} {direction} {value} for {days} days")
    if direction == '>':
        condition = col(indicator) > value
    else:
        condition = col(indicator) < value

    window_spec = Window.partitionBy().orderBy("date").rowsBetween(-days + 1, 0)
    rolling_condition = when(spark_min(when(condition, 1).otherwise(0)).over(window_spec) == 1, 1).otherwise(0)
    return rolling_condition

def evaluate_branch(branch: str, spark: SparkSession) -> DataFrame:
    print(f"Evaluating branch: {branch}")
    parts = branch.split('-')
    indicator, period, indicator_ticker, direction, threshold, days, trading_ticker = parts

    # Load data for indicator and trading
    indicator_df = load_data(spark, indicator_ticker)
    trading_df = load_data(spark, trading_ticker)

    # Rename columns to avoid conflicts
    indicator_df = indicator_df.withColumnRenamed('value', f'{indicator}_{period}')

    # Inner join on date
    common_df = indicator_df.join(trading_df, on="date", how="inner")

    # Evaluate the condition
    condition_met = evaluate_condition(common_df, f"{indicator}_{period}", direction, float(threshold), int(days))

    # Create new columns for the condition and returns
    window_spec = Window.orderBy("date")
    common_df = common_df.withColumn("condition_met", condition_met)
    common_df = common_df.withColumn("shifted_condition", lag("condition_met", 1).over(window_spec))
    common_df = common_df.withColumn("trade_returns_day", 
                                     when(col("shifted_condition") == 1, 
                                          (col("close") / lag("close", 1).over(window_spec)) - 1).otherwise(0))

    return common_df

def save_as_parquet(df: DataFrame, branch: str, output_dir: str) -> None:
    print(f"Saving branch {branch} as Parquet...")
    # Add year, month, and branch columns for partitioning
    df = df.withColumn('year', year(col('date')))
    df = df.withColumn('month', month(col('date')))
    df = df.withColumn('branch', lit(branch))

    # Write partitioned Parquet data
    df.write.partitionBy("year", "month", "branch").mode("append").parquet(output_dir)

def process_branch(branch: str, spark: SparkSession, output_dir: str) -> None:
    try:
        result_df = evaluate_branch(branch, spark)
        print(f"Sample data for branch {branch}:")
        result_df.show(10, truncate=False)
        save_as_parquet(result_df, branch, output_dir)
    except Exception as e:
        print(f"Error processing branch {branch}: {str(e)}")

def main() -> None:
    # Create a Spark session
    spark = create_spark_session()

    with open('branches.txt', 'r') as f:
        branches = f.read().splitlines()
        branches = branches[0:1000]

    print(f"Processing {len(branches)} branches")

    output_dir = './output_data_pure_spark'
    os.makedirs(output_dir, exist_ok=True)

    # Process branches in parallel using Spark
    for branch in tqdm(branches, desc="Processing branches", unit="branch"):
        process_branch(branch, spark, output_dir)

    print("Branch evaluation completed.")

if __name__ == "__main__":
    main()
