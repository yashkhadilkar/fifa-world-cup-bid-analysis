# Should We Bid? — FIFA World Cup Hosting Decision Support Tool

**MSBA 405 · Team 1 · UCLA Anderson**

Anay Mehta, Hans Grunwald, Zahid Ahmed, Matheus Kina, Mubarak Alkharafi, Yash Khadilkar

---
## Overview

This project builds a data-driven decision support tool for national governments evaluating whether to bid to host a FIFA World Cup and for FIFA evaluating which bid should to accept. It ingests economic indicator data from the World Bank (WDI) and IMF, scores every country on hosting readiness using an Isolation Forest anomaly detection model, and surfaces historical pre/post hosting impacts through an interactive Tableau dashboard connected to Snowflake.

The full pipeline is orchestrated by Luigi and runs end-to-end with a single command:

```bash
bash run_pipeline.sh
```

This script creates a Dataproc cluster, runs all pipeline tasks, loads Snowflake, and deletes the cluster automatically.

## Architecture
