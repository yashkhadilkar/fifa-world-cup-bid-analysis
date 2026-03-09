"""
Step 1: IngestWDI (PySpark on Dataproc)
---------------------------------------
Pull every available WDI indicator for all countries from the World Bank API.
Filter to 1960 onward. Write to gs://msba405-team-1-data/raw/wdi/ as Parquet.

Usage (submit to Dataproc):
    gcloud dataproc jobs submit pyspark ingest_wdi.py \
        --cluster=msba405-prototype \
        --region=us-central1 \
        --py-files=gs://msba405-team-1-data/scripts/ingest_wdi.py \
        --properties="spark.jars.packages=com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.11"

"""

import wbgapi as wb
import pandas as pd
import time
import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_OUTPUT = "gs://msba405-team-1-data/raw/wdi/"
START_YEAR = 1960
END_YEAR = 2024
BATCH_SIZE = 50          # indicators per API call (wbgapi chunks automatically, but
                         # we batch to checkpoint progress and avoid memory spikes)
RETRY_LIMIT = 3
RETRY_DELAY = 10         # seconds between retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("IngestWDI")

# ---------------------------------------------------------------------------
# 1. Get the full list of WDI indicator codes
# ---------------------------------------------------------------------------
log.info("Fetching WDI indicator catalog from World Bank API...")
all_indicators = []
for s in wb.series.list():
    all_indicators.append(s["id"])
log.info(f"Found {len(all_indicators)} WDI indicators.")

# ---------------------------------------------------------------------------
# 2. Get economy (country) list, exclude aggregates
# ---------------------------------------------------------------------------
log.info("Fetching economy list (countries only, no aggregates)...")
countries = []
for econ in wb.economy.list():
    # aggregate entries have 'aggregate' field = True, or region != ''
    # wbgapi marks aggregates; we keep only countries
    if econ.get("aggregate") is False or econ.get("region", "") != "":
        countries.append(econ["id"])

# Fallback: if the aggregate filter didn't work as expected, just take all
if len(countries) < 50:
    log.warning("Country filter may have been too aggressive, falling back to all economies.")
    countries = [e["id"] for e in wb.economy.list()]

log.info(f"Will pull data for {len(countries)} economies.")

# ---------------------------------------------------------------------------
# 3. Pull data in batches of indicators
# ---------------------------------------------------------------------------
all_frames = []
indicator_batches = [
    all_indicators[i : i + BATCH_SIZE]
    for i in range(0, len(all_indicators), BATCH_SIZE)
]

log.info(f"Pulling data in {len(indicator_batches)} batches of up to {BATCH_SIZE} indicators each.")

for batch_idx, batch in enumerate(indicator_batches):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            log.info(
                f"Batch {batch_idx + 1}/{len(indicator_batches)} "
                f"({len(batch)} indicators), attempt {attempt}..."
            )

            # wbgapi returns a wide DataFrame: rows = economies, cols = YRxxxx
            # We request all countries and all years in one shot per batch.
            df_wide = wb.data.DataFrame(
                batch,
                economy=countries,
                time=range(START_YEAR, END_YEAR + 1),
                labels=False,       # use codes, not labels (faster)
                numericTimeKeys=True,
                columns="time",     # years as columns
            )

            if df_wide is None or df_wide.empty:
                log.warning(f"Batch {batch_idx + 1} returned empty, skipping.")
                break

            # df_wide index = (economy, series), columns = year integers
            # Melt to long format: economy, series, year, value
            df_wide = df_wide.reset_index()

            # The index after reset has 'economy' and 'series' columns
            year_cols = [c for c in df_wide.columns if isinstance(c, int)]
            df_long = df_wide.melt(
                id_vars=["economy", "series"],
                value_vars=year_cols,
                var_name="year",
                value_name="value",
            )

            # Drop nulls (the API returns NaN for missing observations)
            df_long = df_long.dropna(subset=["value"])

            if not df_long.empty:
                all_frames.append(df_long)
                log.info(
                    f"  -> Got {len(df_long):,} observations "
                    f"({df_long['series'].nunique()} indicators, "
                    f"{df_long['economy'].nunique()} economies)."
                )

            break  # success, move to next batch

        except Exception as e:
            log.error(f"  Batch {batch_idx + 1} attempt {attempt} failed: {e}")
            if attempt < RETRY_LIMIT:
                log.info(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"  Giving up on batch {batch_idx + 1} after {RETRY_LIMIT} attempts.")

    # Small delay between batches to be kind to the API
    time.sleep(1)

# ---------------------------------------------------------------------------
# 4. Concatenate all batches into one pandas DataFrame
# ---------------------------------------------------------------------------
if not all_frames:
    log.error("No data was retrieved. Exiting.")
    sys.exit(1)

log.info("Concatenating all batches...")
pdf = pd.concat(all_frames, ignore_index=True)

# Rename columns to match our schema
pdf.columns = ["iso3", "indicator_code", "year", "value"]
pdf["year"] = pdf["year"].astype(int)
pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
pdf = pdf.dropna(subset=["value"])

log.info(
    f"Final WDI dataset: {len(pdf):,} rows, "
    f"{pdf['indicator_code'].nunique()} indicators, "
    f"{pdf['iso3'].nunique()} economies, "
    f"years {pdf['year'].min()}-{pdf['year'].max()}."
)

# ---------------------------------------------------------------------------
# 5. Convert to Spark DataFrame and write to GCS as Parquet
# ---------------------------------------------------------------------------
log.info("Initializing Spark session...")
spark = SparkSession.builder \
    .appName("IngestWDI") \
    .getOrCreate()

schema = StructType([
    StructField("iso3", StringType(), False),
    StructField("indicator_code", StringType(), False),
    StructField("year", IntegerType(), False),
    StructField("value", DoubleType(), True),
])

sdf = spark.createDataFrame(pdf, schema=schema)

log.info(f"Writing {sdf.count():,} rows to {GCS_OUTPUT} ...")
sdf.write.mode("overwrite").parquet(GCS_OUTPUT)

log.info("IngestWDI complete.")
spark.stop()
