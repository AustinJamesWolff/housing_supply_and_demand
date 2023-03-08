import os
from pyspark.sql.functions import col as spark_col

def read_df(spark, start_year, end_year, src_file):

    # Connect to S3
    # s3client = boto3.client("s3")
    os.environ.setdefault("AWS_PROFILE", "default")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-1")

    # Get file path for spark readings
    src_filepath = "s3://" + src_file

    # READ IN DATAFRAME
    spark_s3_df = spark.read.load(src_filepath, format="csv", header=True)

    # CHANGE TYPE FOR THE YEAR COLUMNS
    for i in range(start_year, end_year + 1):
        spark_s3_df = spark_s3_df.withColumn(
            str(i), spark_col(str(i)).cast("float")
        )
        
    # CHNAGE TYPE FOR THE CROSSWALK MULTIPLIER COLUMNS
    spark_s3_df = spark_s3_df.withColumn('wt_pop', spark_col('wt_pop').cast("float")). \
        withColumn('wt_hu', spark_col('wt_hu').cast("float")). \
        withColumn('wt_adult', spark_col('wt_adult').cast("float")). \
        withColumn('wt_fam', spark_col('wt_fam').cast("float")). \
        withColumn('wt_hh', spark_col('wt_hh').cast("float")). \
        withColumn('parea', spark_col('parea').cast("float"))

    # Drop unnecessary columns
    spark_s3_df = spark_s3_df.drop("name","block group")

    return spark_s3_df
