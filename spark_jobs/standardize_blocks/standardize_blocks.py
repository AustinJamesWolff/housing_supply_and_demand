import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql.functions import col as spark_col

# Import other functions from module
from util import get_spark_session
from read import read_df
from transform import run_standardize_script
from write import write_file

# AWS library
import awswrangler as wr

def main():
    # Get environment variables
    env = os.environ.get(
        "ENVIRON")
    start_year = int(os.environ.get(
        "START_YEAR"))
    end_year = int(os.environ.get(
        "END_YEAR"))
    src_file = os.environ.get(
        "SRC_FILE")
    tgt_file = os.environ.get(
        "TGT_FILE")
    use_weight = os.environ.get(
        "WEIGHT")

    # Connect to S3
    os.environ.setdefault("AWS_PROFILE", "default")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-1")

    # Start the spark session
    spark = get_spark_session(
        env, 'Standardize Block Groups')

    # Read in the Spark Dataframe    
    spark_df = read_df(
        spark, start_year, end_year, src_file)

    # Standardize the data
    standardized_df = run_standardize_script(
        spark_df, env, start_year, end_year, use_weight, src_file)

    # Print the dataframe just to double-check
    standardized_df.show(5)

    # Write to S3 bucket
    write_file(standardized_df, tgt_file)

if __name__ == '__main__':
    main()
