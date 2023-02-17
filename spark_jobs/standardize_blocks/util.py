from pyspark.sql import SparkSession

def get_spark_session(env, app_name):

    print(f"env: {env}")

    if env == "LOCAL":

        # Using findspark ensures the spark path is
        # found and set, allowing us to use the
        # appropriate JAR files to connect to AWS.
        # Only use locally.
        import findspark
        location = findspark.find()
        findspark.init(location, edit_rc=True)

        # Create Spark Session
        spark = SparkSession. \
            builder. \
            master('local'). \
            config("spark.driver.bindAddress","127.0.0.1"). \
            config("spark.driver.host","127.0.0.1"). \
            config("spark.jars.packages", 
                "org.apache.hadoop:hadoop-aws:2.10.2,org.apache.hadoop:hadoop-client:2.10.2"). \
            config("spark.jars.excludes", 
                "com.google.guava:guava"). \
            appName(app_name). \
            getOrCreate()

        # Use the system's default AWS credentials
        spark.sparkContext._jsc.hadoopConfiguration().set(
            "fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
        )

    elif env == "PROD":

        # Create Spark Session
        spark = SparkSession. \
            builder. \
            master('yarn'). \
            appName(app_name). \
            getOrCreate()

    return spark