import awswrangler as wr
import pandas as pd
import numpy as np
import boto3
import requests
from pyspark.sql.types import *
from pyspark.sql.functions import col as spark_col

def run_standardize_script(
    spark_df, env, start_year, end_year, use_weight, src_file):

    # If local, only run a portion of the blocks
    if env == "LOCAL":

        # Test getting unique blocks
        spark_key_list = [i['BG20'] for i in spark_df.select('BG20').limit(10).collect()]

        # CONTINUE TEST: only include blocks in our limited list
        spark_df = spark_df.filter(spark_col('BG20').isin(spark_key_list))

    # Group dataframe by BG20
    spark_groups = spark_df.groupby("BG20")

    # Here are details about accessing cached files:
    # https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-input-distributed-cache.html

    # Call in raw data from cached hadoop file system
    og_df = pd.read_csv("population_blocks_raw.csv",
        encoding='utf-8',
        dtype={'geo_id':str, 'state':str, 'county':str, 
            'tract':str, 'block':str, 'BG10':str, 'BG20':str,
            'TRACT20':str, 'TRACT10':str})
    print(og_df.head(5))

    def block_standardize_spark(key, pdf):

        # Get variables
        block = key[0]
        years_10_19 = [str(i) for i in range(start_year, 2020)]
        years_2020_on = [str(i) for i in range(2020, end_year + 1)]
        
        # Pandas Dataframe version
        df = og_df

        # Step 1: Get a dataframe grouped by BG20
        bg20_df = pdf.drop_duplicates().copy()
        bg20_df = bg20_df.fillna(0)

        # Step 2: Get dot product of 2010-2019 values with the target weight values values
        array_10_19 = bg20_df[years_10_19].to_numpy().T
        wt_array = bg20_df[use_weight].to_numpy()
        dots = array_10_19.dot(wt_array)

        # Step 3: Append standardized 2010-2019 and 
        # the block's 2020 value to new dictionary
        filtered = df[df['block']==block]
        val_20_on = filtered[years_2020_on].iloc[0]
        dots = np.append(dots, val_20_on)

        # If the 2020 value is 0, then all years
        # before then should be 0 also
        if val_20_on[0] == 0:
            dots = dots * 0
            
        # Convert to tuple
        tuple_dots = tuple(dots)

        return pd.DataFrame([key + tuple_dots])

    # Create schema for the key and as many years as we have
    schema2 = StructType(
        [StructField('geoid_block', StringType(), True)] + 
        [StructField(str(i), DoubleType(), True) 
        for i in range(start_year, end_year + 1)])

    # Run standardization function
    spark_standardized = spark_groups.applyInPandas(
        block_standardize_spark, schema=schema2)

    return spark_standardized
