# Should We Accept the Bid? — FIFA World Cup Hosting Decision Support Tool

**MSBA 405 · Team 1 · UCLA Anderson**

---
## Overview

This project builds a data-driven decision support tool for the FIFA World Cup hosting committee. It ingests economic indicator data from the World Bank (WDI) and IMF, evaluates countries across key development indicators using machine learning, and surfaces historical pre/post hosting impacts through an interactive Tableau dashboard connected to Snowflake. The tool helps the committee assess which candidate nations have economic profiles most consistent with successful past hosts.

The full pipeline is orchestrated by Luigi and runs end-to-end with a single command:

```bash
bash run_pipeline.sh
```

This script creates a Dataproc cluster, runs all pipeline tasks, loads Snowflake, and deletes the cluster automatically.

## Pipeline
![Demo](assets/pipeline_architecture.png)

The pipeline flows left to right through five stages:

**Data Sources (GCS):** World Bank WDI (8.4M rows), IMF API (553K rows), and a curated FIFA hosts CSV (33 entries) are stored as Parquet and CSV in Google Cloud Storage.

**Processing (Dataproc):** Two PySpark jobs run on a Dataproc cluster. BuildTrainingFeatures joins WDI + IMF + hosts, computes 6-year pre-event averages, pivots to wide format, and filters correlated indicators at a 0.90 threshold (output: 33 rows, ~177 indicators). BuildEventWindow builds indicator time series from t-6 to t+6 relative to each host's tournament year (output: 172,249 rows).

**ML Model (local):** TrainAndPredict trains a SVM Model (scikit-learn) on host country profiles and scores all countries on hosting similarity (output: 208 country scores).

**Serving (Snowflake):** LoadSnowflake loads three fact tables into the FACTS schema with atomic rollback: if any table fails, all tables from that run are dropped. LoadDimensions then populates the DIMENSIONS schema with DIM_COUNTRY and DIM_INDICATOR using the World Bank API for human-readable names.

**Visualization (Tableau):** A Tableau dashboard connects to Snowflake and provides interactive exploration of hosting similarity and historical impact.


## Data Download and Ingestion

Raw data must be present in GCS before running the pipeline. If the bucket is already populated (as it is for the demo video), skip this section. 

### 1. World Bank WDI Data

Run the WDI ingestion script `ingest_wdi.py` in Colab (this takes ~4 hours and produces 8.4M rows):


### 2. IMF Data

Run the WDI ingestion script `ingest_imf.py` in Colab:

### 3. FIFA Hosts CSV

```bash
gsutil cp fifa_wc_hosts.csv gs://msba405-team-1-data/raw/fifa_wc_hosts.csv
```

### 4. PySpark Scripts

```bash
gsutil cp build_training_features.py gs://msba405-team-1-data/scripts/build_training_features.py
gsutil cp build_event_window.py gs://msba405-team-1-data/scripts/build_event_window.py
```

⭐ All of these scripts should be in a separate scripts folder that should be created in GCS bucket to store all of the scripts for the data sources and processing phase, including `ingest_imf.py`, `ingest_wdi.py`, `build_event_windows.py`, and `build_training_features.py`.

## GCS Bucket Structure

```
gs://msba405-team-1-data/
├── raw/
│   ├── wdi/wdi_data.parquet              
│   ├── imf/imf_data.parquet              
│   └── fifa_wc_hosts.csv                 
├── scripts/
│   ├── build_training_features.py
│   └── build_event_window.py
├── processed/                          
│   ├── training_features/             
│   └── predictions/
│       └── country_scores.parquet       
└── event_window/                       
    └── fact_host_event_window/          
```


## Quick Start

### 1. Prerequisites

**Software (install before running):**

```bash
pip install luigi snowflake-connector-python gcsfs pyarrow pandas numpy scikit-learn
```

You also need the Google Cloud SDK (`gcloud` CLI) authenticated to the `msba405-team-1` project:

```bash
gcloud auth login
gcloud config set project msba405-team-1
```

### 2. One-Time Snowflake Setup

If this is the first time running the pipeline on a fresh Snowflake account, grant permissions on the schemas:
```python
import snowflake.connector

conn = snowflake.connector.connect(
    account="your_account",
    user="your_username",
    password="your_password",
    database="FIFA_WC",
    warehouse="WC_WH",
    role="ACCOUNTADMIN",
)
cur = conn.cursor()

cur.execute("GRANT ALL PRIVILEGES ON SCHEMA FACTS TO ROLE PUBLIC")
cur.execute("GRANT ALL PRIVILEGES ON SCHEMA DIMENSIONS TO ROLE PUBLIC")
cur.execute("GRANT ALL PRIVILEGES ON SCHEMA ANALYTICS TO ROLE PUBLIC")

cur.close()
conn.close()
```

This only needs to be done once.


### 3. Set Snowflake Credentials

Export your Snowflake credentials in your terminal before running:
```bash
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_username"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_WAREHOUSE="WC_WH"
export SNOWFLAKE_DATABASE="FIFA_WC"
export SNOWFLAKE_ROLE="ACCOUNTADMIN"
```

These stay in your terminal session only and are never committed to the repo. See `env.example` for a reference template.
### 3. Clear failed flags

Clear any failed flags if running the pipeline again.

```bash
!rm -f /tmp/luigi_*.flag
```


### 4. Verify Raw Data

The raw data must already be present in GCS. The `run_pipeline.sh` script checks for these files and will exit with a clear error if any are missing. See the "Data Download and Ingestion" section above if you need to populate the bucket from scratch.

### 5. Run the Pipeline

```bash
bash run_pipeline.sh
```

This single command:
1. Validates GCP authentication and Snowflake credentials
2. Clears any failed flags from previous pipeline run
3. Verifies raw data and scripts exist in GCS
4. Creates a Dataproc cluster (`msba405-prototype`)
5. Runs the full Luigi pipeline (sensors, Spark jobs, ML training, Snowflake load)
6. Deletes the Dataproc cluster (even if the pipeline fails, to save credits)

Expected runtime: ~25-30 minutes (cluster creation takes 2-3 min, Spark jobs ~10 min, model + Snowflake ~12-13 min).

⭐ Demo was run in Google Colab


## Fault Tolerance

The pipeline implements fault tolerance at two levels:

**Luigi task-level:** Each task writes a flag file on success. If a task fails, Luigi stops downstream tasks but preserves completed work. Re-running the pipeline skips already-completed tasks (idempotent).

**Snowflake atomic rollback:** The LoadSnowflake task wraps all three table loads in a single try/except block. If any table fails to load, all tables that were successfully created during that run are dropped (rolled back), preventing Snowflake from being left in an inconsistent state with partial data. On re-run, all three tables are reloaded from scratch.

**Cluster cleanup:** The `run_pipeline.sh` script always deletes the Dataproc cluster, even if the pipeline fails. This prevents orphaned clusters from burning credits.

## Snowflake Schema

**Account:** your_account (GCP us-central1)
**Database:** FIFA_WC
**Schemas:** FACTS, DIMENSIONS
**Warehouse:** WC_WH (XSMALL, auto-suspend 60s)

### FACTS.FACT_HOST_EVENT_WINDOW

Indicator time series for each past host country, aligned to event time (years relative to hosting year). 

```sql
CREATE TABLE FACTS.FACT_HOST_EVENT_WINDOW (
    iso3            VARCHAR,     -- Country ISO3 code (e.g. 'BRA', 'DEU')
    indicator_code  VARCHAR,     -- WDI/IMF indicator code (e.g. 'NY.GDP.MKTP.KD.ZG')
    year            INTEGER,     -- Calendar year
    value           FLOAT,       -- Indicator value
    host_iso3       VARCHAR,     -- Host country ISO3 code
    tournament_year INTEGER,     -- FIFA World Cup year
    event_time      INTEGER      -- Year relative to hosting (0 = host year, -6 to +6)
);
-- 172,249 rows
```

### FACTS.COUNTRY_SCORES

Model output scoring every country.

```sql
CREATE TABLE FACTS.COUNTRY_SCORES (
    iso3                    VARCHAR,  -- Country ISO3 code
    anomaly_label           INTEGER,  -- 1 = similar to past hosts, -1 = anomaly
    data_completeness       FLOAT,    -- Fraction of indicators available (0.0 to 1.0)
    hosting_readiness_score FLOAT     -- Normalized score (0 to 100)
);
-- 208 rows
```

### FACTS.TRAINING_FEATURES

Wide-format feature matrix used to train the model. One row per host country per tournament, with ~177 indicator columns (6-year pre-event averages).

```sql
-- Created dynamically via INFER_SCHEMA (column names contain dots from WDI indicator codes)
-- Key columns: "host_iso3" VARCHAR, "tournament_year" NUMBER
-- Feature columns: "AG.CON.FERT.PT.ZS", "BX.KLT.DINV.WD.GD.ZS", etc.
-- All column names are lowercase and quoted due to dots in WDI codes
-- 33 rows
```

### DIMENSIONS.DIM_COUNTRY

Country metadata for joining to fact tables.

```sql
CREATE TABLE DIMENSIONS.DIM_COUNTRY (
    iso3          VARCHAR,    -- Country ISO3 code
    country_name  VARCHAR,    -- Full country name
    region        VARCHAR,    -- World Bank region code (e.g. 'LCN', 'ECS', 'MEA')
    income_group  VARCHAR     -- Income classification (HIC, UMC, LMC, LIC)
);
-- 266 rows
```

### DIMENSIONS.DIM_INDICATOR

Indicator metadata mapping codes to readable names.

```sql
CREATE TABLE DIMENSIONS.DIM_INDICATOR (
    indicator_code VARCHAR,   -- WDI/IMF indicator code
    indicator_name VARCHAR,   -- Human-readable name
    source         VARCHAR    -- Data source ('WDI' or 'IMF')
);
-- 1,541 rows
```

### Snowflake Infrastructure Objects

```sql
-- Schemas
CREATE SCHEMA IF NOT EXISTS FACTS;
CREATE SCHEMA IF NOT EXISTS DIMENSIONS;

-- Storage integration (connects Snowflake to GCS)
CREATE STORAGE INTEGRATION GCS_INTEGRATION
    TYPE = EXTERNAL_STAGE
    STORAGE_PROVIDER = 'GCS'
    ENABLED = TRUE
    STORAGE_ALLOWED_LOCATIONS = ('gcs://msba405-team-1-data/');

-- External stage and file format (in ANALYTICS schema, referenced by pipeline)
CREATE FILE FORMAT ANALYTICS.PARQUET_FF TYPE = PARQUET;

CREATE STAGE ANALYTICS.GCS_STAGE
    STORAGE_INTEGRATION = GCS_INTEGRATION
    URL = 'gcs://msba405-team-1-data/'
    FILE_FORMAT = ANALYTICS.PARQUET_FF;
```

## Tableau Dashboard

**Dashboard link:** [TODO: paste Tableau link here]

The Tableau dashboard connects to Snowflake and provides interactive views for exploring ...

## Project Structure

```
├── run_pipeline.sh              # Single-command entry point (creates cluster, runs pipeline, deletes cluster)
├── pipeline.py                  # Luigi pipeline (task definitions and orchestration)
├── luigi.cfg                    # Luigi configuration
├── build_training_features.py   # PySpark script for training feature extraction
├── build_event_window.py        # PySpark script for event window construction
├── ingest_wdi.py                # WDI data ingestion script (run separately, ~4 hours)
├── ingest_imf.py                # IMF data ingestion script (run separately)
├── fifa_wc_hosts.csv            # Curated host country list (1930-2034)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes credentials, data files, and caches
└── README.md                    # This file
└── Pipeline_Model.ipynb         # One-Class SVM (OC-SVM) model: an unsupervised machine learning algorithm primarily used for anomaly (or outlier) detection.

```

## Team

| Name              | Role                    |
|-------------------|-------------------------|
| Yash Khadilkar    | Technical, ETL pipeline           |
| Anay Mehta        | Technical, Model         |
| Hans Grunwald     | Data curation, dashboard |
| Matheus Kina      | Data curation, dashboard |
| Zahid Ahmed       | Data curation, presentation|
| Mubarak Alkharafi | Data curation, presentation |

## Notes

- All GCP resources are in us-central1 / us-central1-b to avoid data transfer charges.
- The Dataproc cluster is created and deleted by `run_pipeline.sh` automatically.
- The SVM model uses one-class anomaly detection because training data contains only positive examples (past host countries) with no labeled negative examples.
- WDI ingestion is excluded from the Luigi pipeline because it takes ~4 hours. It is treated as a pre-existing GCS asset.
