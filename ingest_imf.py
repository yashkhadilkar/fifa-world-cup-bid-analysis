"""
Step 2: IngestIMF (PySpark on Dataproc)
---------------------------------------
Pull all IMF World Economic Outlook (WEO) indicators for all countries.
Harmonize country codes to ISO3 during ingestion.
Write to gs://msba405-team-1-data/raw/imf/ as Parquet.

The IMF DataMapper API provides ~45 macro indicators (GDP, inflation,
unemployment, current account, government debt, etc.) for 190+ countries
with annual data and forecasts.

Usage (submit to Dataproc):
    gcloud dataproc jobs submit pyspark ingest_imf.py \
        --cluster=msba405-prototype \
        --region=us-central1

Or run locally in Colab:
    !pip install pyspark requests
    %run ingest_imf.py
"""

import requests
import pandas as pd
import time
import logging
import sys
import json

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_OUTPUT = "gs://msba405-team-1-data/raw/imf/"
IMF_BASE_URL = "https://www.imf.org/external/datamapper/api/v1"
RETRY_LIMIT = 3
RETRY_DELAY = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("IngestIMF")

# ---------------------------------------------------------------------------
# Helper: fetch JSON from IMF DataMapper API with retries
# ---------------------------------------------------------------------------
def fetch_imf_json(url):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"  Attempt {attempt} for {url} failed: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                raise

# ---------------------------------------------------------------------------
# 1. Get the full list of available indicators
# ---------------------------------------------------------------------------
log.info("Fetching IMF indicator catalog...")
indicators_json = fetch_imf_json(f"{IMF_BASE_URL}/indicators")
indicator_ids = list(indicators_json.get("indicators", {}).keys())
log.info(f"Found {len(indicator_ids)} IMF indicators.")

# Store indicator metadata (id -> label) for reference
indicator_meta = {}
for ind_id, meta in indicators_json.get("indicators", {}).items():
    indicator_meta[ind_id] = meta.get("label", ind_id)

# ---------------------------------------------------------------------------
# 2. Get country list and build ISO3 mapping
# ---------------------------------------------------------------------------
log.info("Fetching IMF country list...")
countries_json = fetch_imf_json(f"{IMF_BASE_URL}/countries")
# The IMF API uses its own country codes; we need to map to ISO3.
# The countries endpoint returns {code: {label: "Country Name"}}
imf_countries = countries_json.get("countries", {})
log.info(f"Found {len(imf_countries)} IMF country/group codes.")

# IMF DataMapper country codes are typically ISO3 already for countries
# (e.g., "USA", "BRA", "DEU") but also include groups like "ADVEC", "OEMDC".
# We will pull all and filter to ISO3 codes (3 uppercase letters, no digits)
# after pulling data. This is safe because group codes like "ADVEC" or
# "OEMDC" won't match ISO3 patterns used by WDI.
import re
iso3_pattern = re.compile(r"^[A-Z]{3}$")

# ---------------------------------------------------------------------------
# 3. Pull data for each indicator
# ---------------------------------------------------------------------------
all_rows = []

for idx, ind_id in enumerate(indicator_ids):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            log.info(
                f"Indicator {idx + 1}/{len(indicator_ids)}: {ind_id} "
                f"({indicator_meta.get(ind_id, '?')})..."
            )

            # The DataMapper API returns all countries and years for one indicator:
            # {values: {INDICATOR: {COUNTRY: {YEAR: value, ...}, ...}}}
            data_json = fetch_imf_json(f"{IMF_BASE_URL}/{ind_id}")

            values = data_json.get("values", {}).get(ind_id, {})

            if not values:
                log.warning(f"  No data for {ind_id}, skipping.")
                break

            # Parse the nested dict into rows
            for country_code, year_data in values.items():
                if not isinstance(year_data, dict):
                    continue
                for year_str, val in year_data.items():
                    try:
                        year_int = int(year_str)
                        val_float = float(val)
                        all_rows.append({
                            "iso3": country_code,
                            "indicator_code": ind_id,
                            "year": year_int,
                            "value": val_float,
                        })
                    except (ValueError, TypeError):
                        continue

            log.info(f"  -> Parsed {ind_id}, running total: {len(all_rows):,} rows.")
            break  # success

        except Exception as e:
            log.error(f"  Indicator {ind_id} attempt {attempt} failed: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"  Giving up on {ind_id}.")

    # Small delay to respect rate limits
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# 4. Build pandas DataFrame
# ---------------------------------------------------------------------------
if not all_rows:
    log.error("No data was retrieved. Exiting.")
    sys.exit(1)

pdf = pd.DataFrame(all_rows)
log.info(f"Raw IMF data: {len(pdf):,} rows.")

# Filter to likely ISO3 country codes (drop IMF group aggregates)
# ISO3 codes: 3 uppercase letters. IMF groups have mixed patterns.
mask_iso3 = pdf["iso3"].str.match(iso3_pattern)
n_before = len(pdf)
pdf = pdf[mask_iso3].copy()
log.info(f"After filtering to ISO3 codes: {len(pdf):,} rows (dropped {n_before - len(pdf):,} aggregate rows).")

# Some IMF codes are ISO3 but for aggregates (e.g., "EUR" for Euro area).
# We keep them for now; downstream steps can filter if needed.

log.info(
    f"Final IMF dataset: {len(pdf):,} rows, "
    f"{pdf['indicator_code'].nunique()} indicators, "
    f"{pdf['iso3'].nunique()} economies, "
    f"years {pdf['year'].min()}-{pdf['year'].max()}."
)

# ---------------------------------------------------------------------------
# 5. Also save indicator metadata as a small reference table
# ---------------------------------------------------------------------------
meta_rows = [
    {"indicator_code": k, "indicator_label": v}
    for k, v in indicator_meta.items()
]
meta_pdf = pd.DataFrame(meta_rows)

# ---------------------------------------------------------------------------
# 6. Convert to Spark DataFrame and write to GCS as Parquet
# ---------------------------------------------------------------------------
log.info("Initializing Spark session...")
spark = SparkSession.builder \
    .appName("IngestIMF") \
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

# Write metadata reference table
meta_schema = StructType([
    StructField("indicator_code", StringType(), False),
    StructField("indicator_label", StringType(), True),
])
meta_sdf = spark.createDataFrame(meta_pdf, schema=meta_schema)
meta_output = GCS_OUTPUT.rstrip("/") + "_metadata/"
meta_sdf.write.mode("overwrite").parquet(meta_output)
log.info(f"Wrote indicator metadata to {meta_output}")

log.info("IngestIMF complete.")
spark.stop()
