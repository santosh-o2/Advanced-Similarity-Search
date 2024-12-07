from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    FloatType, LongType
)
import time
import os

spark = SparkSession.builder \
    .appName("AmazonSimilaritySearch") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", LongType(), True)
])

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_pth}")
    print(f"Loading data from: {file_path}")
    df = spark.read.csv(file_path, schema=schema, header=False)
    print("Data loaded successfully!")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    #pre-process..
    return df, None, None


def find_similar_products(df, target_product_id):
    print(f"Finding similar products for product ID: {target_product_id}")
    return df

def main():
    dataset_path = "/Users/santosh/Downloads/ratings.csv"   
    try:
        print("Starting data loading process...")
        df = load_data(dataset_path)
        df.cache()
        print("\nDataset Overview:")
        total_records = df.count()
        print(f"Total records: {total_records}")
        print("\nRatings distribution:")
        df.groupBy("rating").count().orderBy("rating").show()  
        print("\nSample of the data:")
        df.show(5)
        processed_df, product_stats, user_product_matrix = preprocess_data(df)
        target_product_id = "0321732944"
        start_time = time.time()
        similar_products = find_similar_products(processed_df, target_product_id)
        execution_time = time.time() - start_time
        
        print("\nSimilar Products:")
        similar_products.orderBy(col("rating").desc()).show()  
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
