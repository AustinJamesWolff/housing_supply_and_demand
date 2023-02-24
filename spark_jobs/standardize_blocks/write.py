import awswrangler as wr

def write_file(standardized_df, tgt_file):

    ### Change to just having spark save to s3, no converting
    standardized_df \
        .write.format("csv") \
        .option('header','true') \
        .option('encoding','utf-8') \
        .mode("overwrite") \
        .save(tgt_file)

    ### Change to just having spark save to s3, 
    ### no converting (with coalesce)
    # standardized_df \
    #     .coalesce(1) \
    #     .write.format("csv") \
    #     .option('header','true') \
    #     .option('encoding','utf-8') \
    #     .mode("overwrite") \
    #     .save(tgt_file)

    # wr.s3.to_csv(
    #     standardized_df.toPandas(), tgt_file, index=False)