# spark-submit --master yarn --deploy-mode cluster \
--conf "spark.yarn.appMasterEnv.ENVIRON=PROD" --conf "spark.yarn.appMasterEnv.START_YEAR=2013" --conf "spark.yarn.appMasterEnv.END_YEAR=2021" --conf "spark.yarn.appMasterEnv.WEIGHT=wt_pop" --conf "spark.yarn.appMasterEnv.SRC_FILE=real-estate-wolff/census-data/block-groups/raw/population_blocks_raw.csv" --conf "spark.yarn.appMasterEnv.TGT_FILE=s3://real-estate-wolff/census-data/block-groups/standardized/population_blocks_standardized.csv" -cacheFile \
s3://real-estate-wolff/census-data/block-groups/raw/population_blocks_raw.csv#population_blocks_raw.csv --py-files s3://real-estate-wolff/scripts/standardize_functions.zip