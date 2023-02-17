import awswrangler as wr

def write_file(standardized_df, tgt_file):

    wr.s3.to_csv(
        standardized_df.toPandas(), tgt_file, index=False)