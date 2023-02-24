#!/bin/bash

# Install needed libraries
sudo pip3 install pandas==1.3.5 awswrangler==2.19.0 boto3==1.26.72

# Create an ec2-user directory for running pyspark in terminal
# sudo -u hdfs hdfs dfs -mkdir /user/ec2-user
# sudo -u hdfs hdfs dfs -chown -R ec2-user:ec2-user /user/ec2-user

# Add py files to the cluster
# scp -i ~/.ssh/rei_development.cer /Users/WonderWolff/Real_Estate/housing_supply_and_demand/spark_jobs/standardize_blocks/standardize_functions.zip ec2-user@{the cluster DNS}:~
# scp -i ~/.ssh/rei_development.cer /Users/WonderWolff/Real_Estate/housing_supply_and_demand/spark_jobs/standardize_blocks/standardize_blocks.py ec2-user@{the cluster DNS}:~

# LOCAL ONLY: Add environment variables
# export ENVIRON=PROD START_YEAR=2013 END_YEAR=2021 WEIGHT=wt_pop SRC_FILE=real-estate-wolff/census-data/block-groups/raw/population_blocks_raw.csv TGT_FILE=s3://real-estate-wolff/census-data/block-groups/standardized/population_blocks_standardized.csv

# LOCAL ONLY: Copy s3 to cached filesystem
# aws s3 cp s3://real-estate-wolff/census-data/block-groups/raw/population_blocks_raw.csv .
