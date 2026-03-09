"""
Step 4: Build Event Window Tables (fact_host_event_window)
=========================================================
For each past World Cup host, this function creates a [-6, +6] year window around the
hosting year for every indicator in WDI + IMF data.  It computes percentage change deltas relative to a pre-hosting baseline (average of t-3 to t-1).

Input:
  gs://msba405-team-1-data/raw/wdi/wdi_data.parquet
  gs://msba405-team-1-data/raw/imf/imf_data.parquet
  gs://msba405-team-1-data/raw/fifa_wc_hosts.csv

Output:
  gs://msba405-team-1-data/event_window/fact_host_event_window/  (Parquet)

Output schema (long format, one row per host x indicator x event_time):
  iso3              STRING   - country code
  host_year         INT      - the World Cup year
  event_time        INT      - relative year (-6 to +6, 0 = hosting year)
  calendar_year     INT      - actual year (host_year + event_time)
  indicator_code    STRING   - WDI or IMF indicator code
  source            STRING   - 'WDI' or 'IMF'
  value             DOUBLE   - raw indicator value for that year
  baseline_value    DOUBLE   - average of t-3, t-2, t-1 values
  pct_change        DOUBLE   - percent change from baseline

Submit to Dataproc:
  gcloud dataproc jobs submit pyspark build_event_window.py \
    --cluster=msba405-prototype \
    --region=us-central1 \
    --project=msba405-team-1
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ── CONFIG ──────────────────────────────────────────────────────────────
BUCKET = "gs://msba405-team-1-data"
WDI_PATH = f"{BUCKET}/raw/wdi/wdi_data.parquet"
IMF_PATH = f"{BUCKET}/raw/imf/imf_data.parquet"
HOSTS_PATH = f"{BUCKET}/raw/fifa_wc_hosts.csv"
OUTPUT_PATH = f"{BUCKET}/event_window/fact_host_event_window"

WINDOW_MIN = -6
WINDOW_MAX = 6
BASELINE_YEARS = [-3, -2, -1]  # event_time values for baseline average


def main():
    spark = SparkSession.builder \
        .appName("Step4_BuildEventWindow") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load raw data ────────────────────────────────────────────────
    print("Loading WDI data...")
    wdi = spark.read.parquet(WDI_PATH) \
        .select(
            F.col("iso3").cast("string"),
            F.col("indicator_code").cast("string"),
            F.col("year").cast("int"),
            F.col("value").cast("double")
        ) \
        .withColumn("source", F.lit("WDI"))

    print(f"  WDI rows: {wdi.count():,}")

    print("Loading IMF data...")
    imf = spark.read.parquet(IMF_PATH) \
        .select(
            F.col("iso3").cast("string"),
            F.col("indicator_code").cast("string"),
            F.col("year").cast("int"),
            F.col("value").cast("double")
        ) \
        .withColumn("source", F.lit("IMF"))

    print(f"  IMF rows: {imf.count():,}")

    # Union both sources
    indicators = wdi.unionByName(imf)
    print(f"  Combined rows: {indicators.count():,}")

    # ── 2. Load hosts ───────────────────────────────────────────────────
    print("Loading hosts CSV...")
    hosts = spark.read.csv(HOSTS_PATH, header=True, inferSchema=True)
    print(f"  Host records: {hosts.count()}")
    hosts.show(truncate=False)

    # Ensure column types
    hosts = hosts.select(
        F.col("iso3").cast("string"),
        F.col("host_year").cast("int")
    )

    # ── 3. Generate event window skeleton ───────────────────────────────
    # For each host, create rows for event_time -6 through +6
    event_times = spark.createDataFrame(
        [(t,) for t in range(WINDOW_MIN, WINDOW_MAX + 1)],
        ["event_time"]
    )

    # Cross join hosts with event times to get (iso3, host_year, event_time)
    host_window = hosts.crossJoin(event_times) \
        .withColumn("calendar_year", F.col("host_year") + F.col("event_time"))

    print(f"  Event window skeleton rows: {host_window.count()}")

    # ── 4. Join indicators to event window ──────────────────────────────
    # Join on iso3 + calendar_year = year
    event_data = host_window.join(
        indicators,
        on=[
            host_window["iso3"] == indicators["iso3"],
            host_window["calendar_year"] == indicators["year"]
        ],
        how="inner"
    ).select(
        host_window["iso3"],
        host_window["host_year"],
        host_window["event_time"],
        host_window["calendar_year"],
        indicators["indicator_code"],
        indicators["source"],
        indicators["value"]
    )

    print(f"  Event data rows after join: {event_data.count():,}")

    # ── 5. Compute baseline (average of t-3, t-2, t-1) ─────────────────
    baseline = event_data.filter(
        F.col("event_time").isin(BASELINE_YEARS)
    ).groupBy("iso3", "host_year", "indicator_code") \
     .agg(F.avg("value").alias("baseline_value"))

    print(f"  Baseline records: {baseline.count():,}")

    # ── 6. Join baseline back and compute pct_change ────────────────────
    result = event_data.join(
        baseline,
        on=["iso3", "host_year", "indicator_code"],
        how="left"
    ).withColumn(
        "pct_change",
        F.when(
            (F.col("baseline_value").isNotNull()) &
            (F.col("baseline_value") != 0.0),
            ((F.col("value") - F.col("baseline_value")) / F.abs(F.col("baseline_value"))) * 100.0
        ).otherwise(None)
    )

    # ── 7. Final column ordering and write ──────────────────────────────
    result = result.select(
        "iso3",
        "host_year",
        "event_time",
        "calendar_year",
        "indicator_code",
        "source",
        "value",
        "baseline_value",
        "pct_change"
    ).orderBy("iso3", "host_year", "indicator_code", "event_time")

    row_count = result.count()
    indicator_count = result.select("indicator_code").distinct().count()
    host_count = result.select("iso3", "host_year").distinct().count()

    print(f"\n{'='*60}")
    print(f"  Output rows:       {row_count:,}")
    print(f"  Unique indicators: {indicator_count:,}")
    print(f"  Host-year combos:  {host_count}")
    print(f"{'='*60}\n")

    print(f"Writing to {OUTPUT_PATH} ...")
    result.write.mode("overwrite").parquet(OUTPUT_PATH)

    # Quick sanity check and show a sample for one host
    print("\nSample: first host, one indicator, full window")
    sample_host = result.select("iso3", "host_year").distinct().first()
    if sample_host:
        sample_indicator = result.filter(
            (F.col("iso3") == sample_host["iso3"]) &
            (F.col("host_year") == sample_host["host_year"])
        ).select("indicator_code").distinct().first()

        if sample_indicator:
            result.filter(
                (F.col("iso3") == sample_host["iso3"]) &
                (F.col("host_year") == sample_host["host_year"]) &
                (F.col("indicator_code") == sample_indicator["indicator_code"])
            ).orderBy("event_time").show(13, truncate=False)

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
