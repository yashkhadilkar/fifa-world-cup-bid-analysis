"""
FIFA World Cup Hosting Analysis Pipeline
MSBA 405 - Team 1

Luigi-orchestrated pipeline that processes World Bank WDI and IMF data,
builds training features and event window tables on Dataproc (PySpark),
trains an Isolation Forest model locally, and loads results into Snowflake.

Usage (from Google Colab):
    !python pipeline.py RunAll
"""

import luigi
import subprocess
import os
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GCS_BUCKET = "gs://msba405-team-1-data"
GCP_PROJECT = "msba405-team-1"
GCP_REGION = "us-central1"
CLUSTER_NAME = "msba405-prototype"

SNOWFLAKE_ACCOUNT = "LHNMKHH-EG10620"
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_DATABASE = "FIFA_WC"
SNOWFLAKE_SCHEMA = "ANALYTICS"
SNOWFLAKE_WAREHOUSE = "WC_WH"


# ---------------------------------------------------------------------------
# Helper: run a shell command and stream output
# ---------------------------------------------------------------------------
def run_cmd(cmd, description=""):
    """Run a shell command, stream output, and raise on failure."""
    logger.info(f"Running: {description or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout[-2000:])
    if result.returncode != 0:
        logger.error(f"FAILED: {description or cmd}")
        logger.error(result.stderr[-2000:])
        raise RuntimeError(f"Command failed (exit {result.returncode}): {description or cmd}")
    return result.stdout


# ===========================================================================
# Sensors: verify pre-existing data and infrastructure
# ===========================================================================

class CheckWDIData(luigi.ExternalTask):
    """Sensor: confirm WDI parquet exists in GCS (pre-ingested)."""
    def output(self):
        return luigi.LocalTarget("/tmp/luigi_check_wdi.flag")

    def run(self):
        run_cmd(
            f"gsutil ls {GCS_BUCKET}/raw/wdi/wdi_data.parquet",
            "Check WDI data in GCS"
        )
        with self.output().open("w") as f:
            f.write("ok")


class CheckIMFData(luigi.ExternalTask):
    """Sensor: confirm IMF parquet exists in GCS (pre-ingested)."""
    def output(self):
        return luigi.LocalTarget("/tmp/luigi_check_imf.flag")

    def run(self):
        run_cmd(
            f"gsutil ls {GCS_BUCKET}/raw/imf/imf_data.parquet",
            "Check IMF data in GCS"
        )
        with self.output().open("w") as f:
            f.write("ok")


class CheckCluster(luigi.ExternalTask):
    """Sensor: confirm Dataproc cluster is running."""
    def output(self):
        return luigi.LocalTarget("/tmp/luigi_check_cluster.flag")

    def run(self):
        out = run_cmd(
            f"gcloud dataproc clusters describe {CLUSTER_NAME} "
            f"--region={GCP_REGION} --format='value(status.state)'",
            "Check Dataproc cluster status"
        )
        state = out.strip()
        if state != "RUNNING":
            raise RuntimeError(
                f"Cluster {CLUSTER_NAME} is in state '{state}', expected RUNNING. "
                f"Start it with: gcloud dataproc clusters start {CLUSTER_NAME} --region={GCP_REGION}"
            )
        with self.output().open("w") as f:
            f.write("ok")


# ===========================================================================
# Dataproc PySpark tasks
# ===========================================================================

class BuildTrainingFeatures(luigi.Task):
    """
    Step 3: Join WDI + IMF + hosts CSV, compute 6-year pre-event window
    averages, pivot to wide format, apply correlation filter (0.90).
    Output: processed/training_features/ (33 rows x ~177 indicators)
    """
    def requires(self):
        return [CheckWDIData(), CheckIMFData(), CheckCluster()]

    def output(self):
        return luigi.LocalTarget("/tmp/luigi_training_features.flag")

    def run(self):
        run_cmd(
            f"gcloud dataproc jobs submit pyspark "
            f"{GCS_BUCKET}/scripts/build_training_features.py "
            f"--cluster={CLUSTER_NAME} --region={GCP_REGION}",
            "Dataproc: BuildTrainingFeatures"
        )
        # Verify output exists
        run_cmd(
            f"gsutil ls {GCS_BUCKET}/processed/training_features/",
            "Verify training features output"
        )
        with self.output().open("w") as f:
            f.write("ok")


class BuildEventWindow(luigi.Task):
    """
    Step 4: For each past host, create indicator values from t-6 to t+6
    relative to hosting year. Long format for Tableau flexibility.
    Output: event_window/fact_host_event_window/ (172,249 rows)
    """
    def requires(self):
        return [CheckWDIData(), CheckIMFData(), CheckCluster()]

    def output(self):
        return luigi.LocalTarget("/tmp/luigi_event_window.flag")

    def run(self):
        run_cmd(
            f"gcloud dataproc jobs submit pyspark "
            f"{GCS_BUCKET}/scripts/build_event_window.py "
            f"--cluster={CLUSTER_NAME} --region={GCP_REGION}",
            "Dataproc: BuildEventWindow"
        )
        # Verify output exists
        run_cmd(
            f"gsutil ls {GCS_BUCKET}/event_window/fact_host_event_window/",
            "Verify event window output"
        )
        with self.output().open("w") as f:
            f.write("ok")


# ===========================================================================
# Local ML task (scikit-learn Isolation Forest)
# ===========================================================================

class TrainAndPredict(luigi.Task):
    """
    Step 5: Train Isolation Forest on host country profiles (33 rows),
    then score all ~261 countries. Outputs predictions to GCS.
    Runs locally (not on Dataproc) since scikit-learn is used.
    """
    def requires(self):
        return [BuildTrainingFeatures()]

    def output(self):
        return luigi.LocalTarget("/tmp/luigi_predictions.flag")

    def run(self):
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        import pyarrow.parquet as pq
        import gcsfs

        fs = gcsfs.GCSFileSystem(project=GCP_PROJECT)

        # ---- Load training features (host countries) ----
        logger.info("Loading training features from GCS...")
        train_path = f"msba405-team-1-data/processed/training_features/"
        train_df = pq.ParquetDataset(train_path, filesystem=fs).read().to_pandas()

        id_cols = ["host_iso3", "tournament_year"]
        feature_cols = [c for c in train_df.columns if c not in id_cols]

        X_train = train_df[feature_cols].values

        # ---- Impute and scale ----
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)

        # ---- Train Isolation Forest ----
        logger.info("Training Isolation Forest...")
        model = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=42
        )
        model.fit(X_train_scaled)

        # ---- Score all countries ----
        logger.info("Building all-country feature matrix...")

        # Load raw WDI + IMF
        wdi_df = pq.ParquetDataset("msba405-team-1-data/raw/wdi/", filesystem=fs).read().to_pandas()
        imf_df = pq.ParquetDataset("msba405-team-1-data/raw/imf/", filesystem=fs).read().to_pandas()

        combined = pd.concat([wdi_df, imf_df], ignore_index=True)

        # Use 2019-2024 window (anchored to WDI max year)
        max_year = int(wdi_df["year"].max())
        window_start = max_year - 5
        recent = combined[(combined["year"] >= window_start) & (combined["year"] <= max_year)]

        # Pivot to wide: one row per country
        agg = recent.groupby(["iso3", "indicator_code"])["value"].mean().reset_index()
        wide = agg.pivot(index="iso3", columns="indicator_code", values="value")

        # Align columns to training features
        for col in feature_cols:
            if col not in wide.columns:
                wide[col] = np.nan
        wide = wide[feature_cols]

        # Filter: require 70% completeness
        completeness = wide.notna().mean(axis=1)
        wide_filtered = wide[completeness >= 0.70].copy()

        X_all = wide_filtered.values
        X_all_imp = imputer.transform(X_all)
        X_all_scaled = scaler.transform(X_all_imp)

        # Predict
        scores = model.decision_function(X_all_scaled)
        labels = model.predict(X_all_scaled)

        # Normalize scores to 0-100
        s_min, s_max = scores.min(), scores.max()
        normalized = ((scores - s_min) / (s_max - s_min)) * 100

        results = pd.DataFrame({
            "iso3": wide_filtered.index,
            "anomaly_label": labels,
            "data_completeness": completeness[completeness >= 0.70].values,
            "hosting_readiness_score": np.round(normalized, 2)
        })

        # Write predictions to GCS
        logger.info(f"Writing {len(results)} country scores to GCS...")
        out_path = f"msba405-team-1-data/processed/predictions/country_scores.parquet"
        results.to_parquet(f"gs://{out_path}", index=False)

        with self.output().open("w") as f:
            f.write("ok")


# ===========================================================================
# Snowflake load with rollback/atomicity
# ===========================================================================

class LoadSnowflake(luigi.Task):
    """
    Load all three tables into Snowflake from GCS via external stage.
    Implements atomic rollback: if any table fails to load, all tables
    created during this run are dropped to avoid inconsistent state.

    Tables loaded:
      1. FACT_HOST_EVENT_WINDOW (from event_window/)
      2. COUNTRY_SCORES (from processed/predictions/)
      3. TRAINING_FEATURES (from processed/training_features/, via INFER_SCHEMA)
    """
    def requires(self):
        return [BuildEventWindow(), TrainAndPredict()]

    def output(self):
        return luigi.LocalTarget("/tmp/luigi_snowflake.flag")

    def run(self):
        import snowflake.connector

        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            warehouse=SNOWFLAKE_WAREHOUSE,
        )
        cur = conn.cursor()

        # Track which tables were successfully created/loaded in this run
        # so we can roll them back on failure
        loaded_tables = []

        try:
            # ------ Table 1: FACT_HOST_EVENT_WINDOW ------
            logger.info("Loading FACT_HOST_EVENT_WINDOW...")
            cur.execute("DROP TABLE IF EXISTS FACT_HOST_EVENT_WINDOW;")
            cur.execute("""
                CREATE TABLE FACT_HOST_EVENT_WINDOW (
                    iso3           VARCHAR,
                    indicator_code VARCHAR,
                    year           INTEGER,
                    value          FLOAT,
                    host_iso3      VARCHAR,
                    tournament_year INTEGER,
                    event_time     INTEGER
                );
            """)
            cur.execute("""
                COPY INTO FACT_HOST_EVENT_WINDOW
                FROM @GCS_STAGE/event_window/fact_host_event_window/
                FILE_FORMAT = (FORMAT_NAME = PARQUET_FF)
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
                ON_ERROR = 'CONTINUE';
            """)
            row_count = cur.fetchone()[0]
            logger.info(f"  FACT_HOST_EVENT_WINDOW: {row_count} rows loaded")
            loaded_tables.append("FACT_HOST_EVENT_WINDOW")

            # ------ Table 2: COUNTRY_SCORES ------
            logger.info("Loading COUNTRY_SCORES...")
            cur.execute("DROP TABLE IF EXISTS COUNTRY_SCORES;")
            cur.execute("""
                CREATE TABLE COUNTRY_SCORES (
                    iso3                    VARCHAR,
                    anomaly_label           INTEGER,
                    data_completeness       FLOAT,
                    hosting_readiness_score FLOAT
                );
            """)
            cur.execute("""
                COPY INTO COUNTRY_SCORES
                FROM @GCS_STAGE/processed/predictions/
                FILE_FORMAT = (FORMAT_NAME = PARQUET_FF)
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
                PATTERN = '.*country_scores.*'
                ON_ERROR = 'CONTINUE';
            """)
            row_count = cur.fetchone()[0]
            logger.info(f"  COUNTRY_SCORES: {row_count} rows loaded")
            loaded_tables.append("COUNTRY_SCORES")

            # ------ Table 3: TRAINING_FEATURES ------
            logger.info("Loading TRAINING_FEATURES...")
            cur.execute("DROP TABLE IF EXISTS TRAINING_FEATURES;")

            # Use INFER_SCHEMA for wide-format parquet (~177 indicator columns)
            cur.execute("""
                CREATE TABLE TRAINING_FEATURES
                USING TEMPLATE (
                    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
                    FROM TABLE(
                        INFER_SCHEMA(
                            LOCATION => '@GCS_STAGE/processed/training_features/',
                            FILE_FORMAT => 'PARQUET_FF'
                        )
                    )
                );
            """)
            cur.execute("""
                COPY INTO TRAINING_FEATURES
                FROM @GCS_STAGE/processed/training_features/
                FILE_FORMAT = (FORMAT_NAME = PARQUET_FF)
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
                ON_ERROR = 'CONTINUE';
            """)
            # ON_ERROR = CONTINUE to skip 0-byte part files PySpark may leave
            row_count = cur.fetchone()[0]
            logger.info(f"  TRAINING_FEATURES: {row_count} rows loaded")
            loaded_tables.append("TRAINING_FEATURES")

            # ------ All tables loaded successfully ------
            logger.info("All 3 Snowflake tables loaded successfully.")
            with self.output().open("w") as f:
                f.write(json.dumps({
                    "tables": loaded_tables,
                    "status": "success"
                }))

        except Exception as e:
            # ============================================================
            # ROLLBACK: drop any tables that were loaded in this run
            # to avoid leaving Snowflake in an inconsistent state.
            # ============================================================
            logger.error(f"Snowflake load FAILED: {e}")
            logger.error(f"Rolling back {len(loaded_tables)} table(s): {loaded_tables}")

            for table in loaded_tables:
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {table};")
                    logger.info(f"  Rolled back: {table}")
                except Exception as drop_err:
                    logger.error(f"  Failed to drop {table} during rollback: {drop_err}")

            raise RuntimeError(
                f"Snowflake load failed and rolled back {len(loaded_tables)} table(s). "
                f"Original error: {e}"
            )
        finally:
            cur.close()
            conn.close()


# ===========================================================================
# Top-level task
# ===========================================================================

class RunAll(luigi.WrapperTask):
    """Run the full pipeline end-to-end."""
    def requires(self):
        return [LoadSnowflake()]


if __name__ == "__main__":
    luigi.run()
