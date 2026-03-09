# ============================================================
# Step 3: BuildTrainingFeatures
# Run on Dataproc via: gcloud dataproc jobs submit pyspark ...
# ============================================================
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import numpy as np
import pandas as pd

spark = SparkSession.builder.appName("BuildTrainingFeatures").getOrCreate()

BUCKET = "gs://msba405-team-1-data"

# ----------------------------------------------------------
# 1. Read raw data, tag source, union
# ----------------------------------------------------------
wdi = (spark.read.parquet(f"{BUCKET}/raw/wdi/wdi_data.parquet")
       .withColumn("source", F.lit("WDI")))

imf = (spark.read.parquet(f"{BUCKET}/raw/imf/imf_data.parquet")
       .withColumn("source", F.lit("IMF")))

indicators = wdi.unionByName(imf)
print(f"Combined indicators: {indicators.count()} rows")

# ----------------------------------------------------------
# 2. Read host list
# ----------------------------------------------------------
hosts = (spark.read.csv(f"{BUCKET}/raw/fifa_wc_hosts.csv", header=True, inferSchema=True)
         .select(
             F.col("iso3").alias("host_iso3"),
             F.col("year").alias("tournament_year")
         ))

# Deduplicate co-hosts (e.g., 2002 KOR/JPN each get their own row already)
hosts = hosts.dropDuplicates(["host_iso3", "tournament_year"])
print(f"Host-country-year pairs: {hosts.count()}")

# ----------------------------------------------------------
# 3. Build event window features (t-6 to t-1 average)
# ----------------------------------------------------------
# Cross join hosts with indicators on matching iso3,
# then filter to the 6-year pre-event window
windowed = (hosts
    .join(indicators, hosts.host_iso3 == indicators.iso3, "inner")
    .filter(
        (F.col("year") >= F.col("tournament_year") - 6) &
        (F.col("year") <= F.col("tournament_year") - 1)
    )
)

# Average each indicator over the 6-year window per host-tournament
event_features_long = (windowed
    .groupBy("host_iso3", "tournament_year", "indicator_code")
    .agg(
        F.avg("value").alias("avg_value"),
        F.count("value").alias("year_count")
    )
)

print(f"Event features (long): {event_features_long.count()} rows")

# ----------------------------------------------------------
# 4. Pivot to wide format
# ----------------------------------------------------------
event_features_wide = (event_features_long
    .groupBy("host_iso3", "tournament_year")
    .pivot("indicator_code")
    .agg(F.first("avg_value"))
)

num_cols = [c for c in event_features_wide.columns 
            if c not in ("host_iso3", "tournament_year")]
print(f"Wide format: {event_features_wide.count()} rows x {len(num_cols)} indicator columns")

# ----------------------------------------------------------
# 5. Drop columns with >50% nulls
# ----------------------------------------------------------
row_count = event_features_wide.count()
null_threshold = 0.5

# Use backticks to handle dots in column names
null_counts = event_features_wide.select(
    [F.sum(F.when(F.col(f"`{c}`").isNull(), 1).otherwise(0)).alias(c) for c in num_cols]
).collect()[0]

keep_cols = [c for c in num_cols if null_counts[c] / row_count <= null_threshold]
drop_null = [c for c in num_cols if c not in keep_cols]
print(f"Dropping {len(drop_null)} columns with >{null_threshold*100}% nulls, keeping {len(keep_cols)}")

features = event_features_wide.select("host_iso3", "tournament_year", *[f"`{c}`" for c in keep_cols])

# ----------------------------------------------------------
# 6. Correlation filtering (drop one of any pair with |r| > 0.90)
# ----------------------------------------------------------
import pandas as pd

pdf = features.toPandas()
numeric_pdf = pdf[keep_cols].apply(pd.to_numeric, errors='coerce')

corr_matrix = numeric_pdf.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.90)]
print(f"Dropping {len(to_drop_corr)} columns with correlation > 0.90")

final_cols = [c for c in keep_cols if c not in to_drop_corr]
print(f"Final feature set: {len(final_cols)} indicators")

# ----------------------------------------------------------
# 7. Write training features
# ----------------------------------------------------------
final_df = features.select("host_iso3", "tournament_year", *[f"`{c}`" for c in final_cols])

(final_df.write
 .mode("overwrite")
 .parquet(f"{BUCKET}/processed/training_features/"))

print(f"Written to {BUCKET}/processed/training_features/")
print(f"Final shape: {final_df.count()} rows x {len(final_df.columns)} columns")

spark.stop()