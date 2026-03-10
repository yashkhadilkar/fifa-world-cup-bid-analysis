# Should We Bid? — FIFA World Cup Hosting Decision Support Tool

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

## Architecture
> See `architecture_diagram.png` in the repo for a visual diagram.

The pipeline flows left to right through five stages:

**Data Sources (GCS):** World Bank WDI (8.4M rows), IMF API (553K rows), and a curated FIFA hosts CSV (33 entries) are stored as Parquet and CSV in Google Cloud Storage.

**Processing (Dataproc):** Two PySpark jobs run on a Dataproc cluster. BuildTrainingFeatures joins WDI + IMF + hosts, computes 6-year pre-event averages, pivots to wide format, and filters correlated indicators at a 0.90 threshold (output: 33 rows, ~177 indicators). BuildEventWindow builds indicator time series from t-6 to t+6 relative to each host's tournament year (output: 172,249 rows).
