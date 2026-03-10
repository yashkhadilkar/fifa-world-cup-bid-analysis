# Should We Accept the Bid? — FIFA World Cup Hosting Decision Support Tool

**MSBA 405 · Team 1 · UCLA Anderson**

Hans Grunwald, Anay Mehta, Yash Khadilkar, Zahid Ahmed, Matheus Kina, Mubarak Alkharafi

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

### 2. Set Snowflake Credentials

Export your Snowflake credentials in your terminal before running:

```bash
export SNOWFLAKE_ACCOUNT="LHNMKHH-EG10620"
export SNOWFLAKE_USER="your_username"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_WAREHOUSE="WC_WH"
export SNOWFLAKE_DATABASE="FIFA_WC"
export SNOWFLAKE_ROLE="PUBLIC"
```

These stay in your terminal session only and are never committed to the repo.

### 3. Verify Raw Data

The raw data must already be present in GCS. The `run_pipeline.sh` script checks for these files and will exit with a clear error if any are missing. See the "Data Download and Ingestion" section below if you need to populate the bucket from scratch.

### 4. Run the Pipeline

```bash
bash run_pipeline.sh
```

This single command:
1. Validates GCP authentication and Snowflake credentials
2. Verifies raw data and scripts exist in GCS
3. Creates a Dataproc cluster (`msba405-prototype`)
4. Runs the full Luigi pipeline (sensors, Spark jobs, ML training, Snowflake load)
5. Deletes the Dataproc cluster (even if the pipeline fails, to save credits)

Expected runtime: ~10-15 minutes (cluster creation takes 2-3 min, Spark jobs ~5 min, model + Snowflake ~2 min).
