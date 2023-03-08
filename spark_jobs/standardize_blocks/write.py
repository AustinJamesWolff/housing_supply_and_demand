import awswrangler as wr

def write_file(standardized_df, tgt_file):

    # Have spark save to s3
    standardized_df \
        .write.format("csv") \
        .option('header','true') \
        .option('encoding','utf-8') \
        .mode("overwrite") \
        .save(tgt_file)
