from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import col
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import numpy as np
import time

spark = SparkSession.builder \
    .appName("SimilarityMetrics") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

#training data
data = [
   ("AKM1MP6P0OYPR", "0132793040", "Great product", 5.0, 1365811200),
("A2CX7LUOHB2NDG", "0321732944", "Good quality", 5.0, 1341100800),
("A2NWSAGRHCP8N5", "0439886341", "Not satisfied", 1.0, 1367193600),
("A2WNBOD3WNDNKT", "0439886341", "Average product", 3.0, 1374451200),
("A1GI0U4ZRJA8WN", "0439886341", "Terrible product", 1.0, 1334707200),
("A1QGNMC6O1VW39", "0511189877", "Excellent product", 5.0, 1397433600),
("A3J3BRHTDRFJ2G", "0511189877", "Not worth it", 2.0, 1397433600),
("A2TY0BTJOTENPG", "0511189877", "Good value for money", 5.0, 1395878400),
("A34ATBPOK6HCHY", "0511189877", "Really good quality", 5.0, 1395532800),
("A89DO69P0XZ27", "0511189877", "Fantastic purchase", 5.0, 1395446400),
("AZYNQZ94U6VDB", "0511189877", "Very satisfied", 5.0, 1401321600),
("A1DA3W4GTFXP6O", "0528881469", "Highly recommended", 5.0, 1405641600),
("A29LPQQDG7LD5J", "0528881469", "Poor quality", 1.0, 1352073600),
("AO94DHGC771SJ", "0528881469", "Very good product", 5.0, 1370131200),
("AMO214LNFCEI4", "0528881469", "Not good at all", 1.0, 1290643200),
("A28B1G1MSJ6OO1", "0528881469", "Great value for money", 4.0, 1280016000),
("A3N7T0DY83Y4IG", "0528881469", "Okay product", 3.0, 1283990400),
("A1H8PY3QHMQQA0", "0528881469", "Not up to the mark", 2.0, 1290556800),
("A2CPBQ5W4OGBX", "0528881469", "Could be better", 2.0, 1277078400),
("A265MKAR2WEH3Y", "0528881469", "Very good", 4.0, 1294790400),
("A37K02NKUIT68K", "0528881469", "Perfect", 5.0, 1293235200),
("A2AW1SSVUIYV9Y", "0528881469", "Great product", 4.0, 1289001600),
("A2AEHUKOV014BP", "0528881469", "Excellent quality", 5.0, 1284249600),
("AMLFNXUIEMN4T", "0528881469", "Terrible product", 1.0, 1307836800),
("A2O8FIJR9EBU56", "0528881469", "Good product", 4.0, 1278547200),
("A3IQGFB959IR4P", "0528881469", "Not happy with the product", 1.0, 1327363200),
("AYTBGUX49LF3W", "0528881469", "Worth the price", 4.0, 1398470400),
("A24QFSUU00IZ05", "0528881469", "Disappointing", 2.0, 1293580800),
("A1NG5X8VYZWX0Q", "0528881469", "Very bad experience", 1.0, 1286582400),
("A1E4WG8HRWWK4R", "0528881469", "Excellent product", 5.0, 1390867200),
("A2AOEW5UGXFOOQ", "0528881469", "Perfect quality", 5.0, 1294790400),
("A2XSWV6AQI90BR", "0528881469", "Worst product ever", 1.0, 1299974400),
("AR84FMFYCQCWF", "0528881469", "Really bad", 1.0, 1296691200),
("A19TBA1WARJS55", "0528881469", "Not bad", 2.0, 1340496000),
("A3C5SMBSKKWNPT", "0528881469", "Good product", 5.0, 1285113600),
("A24EV6RXELQZ63", "0528881469", "Really bad", 1.0, 1317254400),
("A3T6ZQONABIJSG", "0528881469", "Waste of money", 1.0, 1358035200),
("A132P6YSJSI5G2", "0528881469", "Could be better", 2.0, 1298851200),
("A1NQPG5IJ43HJI", "0558835155", "Okay product", 3.0, 1372550400),
("A2WOJCFAWI8VS8", "059400232X", "Great product", 5.0, 1379376000),
("A22FB2WSZSXSHH", "059400232X", "Good product", 5.0, 1348790400),
("AZQZ3STMCBG5H", "059400232X", "Excellent", 5.0, 1395705600),
("A2EGPA22UHMQXL", "0594012015", "Very poor quality", 1.0, 1405296000),
("AC57CU3TF6ZMJ", "0594012015", "Great product", 5.0, 1404172800),
("A2YX0Z6RHA8Y2H", "0594012015", "Very bad", 1.0, 1384300800),
("A38T51B7J6QVD9", "0594012015", "Awful", 1.0, 1377129600),
("A1HOSS7PNC1LMU", "0594012015", "Terrible experience", 1.0, 1374537600),
("A9HI77BE35VYE", "0594012015", "Poor quality", 1.0, 1373932800),
("A3HFUWKSPF5QEH", "0594012015", "Amazing", 5.0, 1374278400),
("A38ETMHM9ATSF0", "0594012015", "Not good", 1.0, 1374105600),
("A34O6MND17VDDH", "0594017343", "Awful", 1.0, 1323216000),
("A3JSAGWSLY5044", "0594017580", "Okay", 3.0, 1404432000),
("A3B12X05JZKM94", "0594033896", "Excellent product", 5.0, 1405382400),
("AKZOMZCNU41GZ", "0594033896", "Good quality", 4.0, 1388620800),
("A2XTFY0GJDF3R2", "0594033896", "Good, but could be better", 4.0, 1300233600),
("A3ILG56NE5QU37", "0594033896", "Best product", 5.0, 1399420800),
("AOK77THMKCGXZ", "0594033896", "Good quality", 4.0, 1394496000),
("A24RBL3W5W6SAN", "0594033926", "Great purchase", 5.0, 1392422400),
("A2RX85Y4S27N2J", "0594033926", "Poor quality", 1.0, 1397865600),
("A253JJFXQNPCOJ", "0594033926", "Good", 4.0, 1387584000),
("AHYURLVH267MA", "0594033926", "Perfect", 5.0, 1389225600),
("A1FXYTO11KEJ2P", "0594033926", "Very good", 4.0, 1398297600),
("A2TKKYL3GKFS2M", "0594033926", "Fantastic product", 5.0, 1388102400),
("A3F069UFW04K1E", "0594033926", "Very good", 5.0, 1392249600),
("A3HQ8RXUMJ2Y58", "0594033926", "Excellent quality", 5.0, 1400976000),
("A2MNKBHPMBB0EA", "0594033926", "Amazing", 5.0, 1390003200),
("A3HGTTFIDLCWDJ", "0594033926", "Good quality", 4.0, 1396483200),
("AYM7NJTTX3M0L", "0594033926", "Fantastic", 5.0, 1393200000),
("A2PL24SVISXZM9", "0594033926", "Perfect product", 5.0, 1392422400)

]

columns = ["user_id", "product_id", "review", "rating", "timestamp"]

df = spark.createDataFrame(data, columns)

#preprocess..
tokenizer = Tokenizer(inputCol="review", outputCol="tokens")
tokens_df = tokenizer.transform(df)
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
filtered_df = remover.transform(tokens_df)
hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features")
tf_df = hashing_tf.transform(filtered_df)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)
print("TF-IDF Output:")
tfidf_df.select("product_id", "review", "tfidf_features").show(truncate=False)
#2-Similarity
def compute_similarity(df, metric):
    start_time = time.time()
    features = df.select("tfidf_features").rdd.map(lambda row: row.tfidf_features.toArray())
    features_matrix = np.array(features.collect())
    if metric == "cosine":
        similarity_matrix = np.dot(features_matrix, features_matrix.T)
    elif metric == "jaccard":
        similarity_matrix = np.array([
            [
                np.sum(np.minimum(a, b)) / np.sum(np.maximum(a, b)) if np.sum(np.maximum(a, b)) != 0 else 0
                for b in features_matrix
            ]
            for a in features_matrix
        ])
    elif metric == "laplacian":
        similarity_matrix = np.array([
            [
                np.exp(-np.sum((a - b) ** 2))
                for b in features_matrix
            ]
            for a in features_matrix
        ])
    elif metric == "euclidean":
        similarity_matrix = np.linalg.norm(features_matrix[:, None] - features_matrix, axis=-1)
    elif metric == "manhattan":
        similarity_matrix = np.abs(features_matrix[:, None] - features_matrix).sum(axis=-1)
    elif metric == "hamming":
        similarity_matrix = np.array([
            [
                np.sum(a != b)
                for b in features_matrix
            ]
            for a in features_matrix
        ])
    else:
        raise ValueError("Invalid metric")

    end_time = time.time()
    return similarity_matrix, end_time - start_time

metrics = ["cosine", "jaccard", "laplacian", "euclidean", "manhattan", "hamming"]
performance = {}

for metric in metrics:
    sim_matrix, duration = compute_similarity(tfidf_df, metric)
    performance[metric] = duration
    print(f"{metric.capitalize()} Similarity Matrix:\n", sim_matrix)

#optimization
tfidf_df.cache()  s
tfidf_df = tfidf_df.repartition(4)  

#report comparission
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(performance.keys(), performance.values(), color="skyblue")
plt.title("Performance Comparison (Bar Chart)")
plt.ylabel("Execution Time (s)")
plt.xlabel("Similarity Metrics")
plt.subplot(1, 2, 2)
plt.pie(performance.values(), labels=performance.keys(), autopct="%1.1f%%", startangle=140)
plt.title("Performance Comparison (Pie Chart)")
plt.tight_layout()
plt.show()
spark.stop()
