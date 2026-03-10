"""
FIFA World Cup Hosting Analysis Pipeline
MSBA 405 — Team 1

Luigi-orchestrated pipeline that processes World Bank WDI and IMF data,
builds training features and event window tables on Dataproc (PySpark),
trains a model locally, and loads results into Snowflake.

Usage:
    export SNOWFLAKE_USER="..." SNOWFLAKE_PASSWORD="..."
    python pipeline.py RunAll --local-scheduler
"""

import luigi
import subprocess
import os
import json
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

SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "LHNMKHH-EG10620")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "FIFA_WC")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "WC_WH")
SNOWFLAKE_ROLE = os.environ.get("SNOWFLAKE_ROLE", "PUBLIC")

# Schemas
FACTS_SCHEMA = "FACTS"
DIMENSIONS_SCHEMA = "DIMENSIONS"


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
                f"Cluster {CLUSTER_NAME} is in state '{state}', expected RUNNING."
            )
        with self.output().open("w") as f:
            f.write("ok")


# ===========================================================================
# Dataproc PySpark tasks
# ===========================================================================

class BuildTrainingFeatures(luigi.Task):
    """
    Join WDI + IMF + hosts CSV, compute 6-year pre-event window
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
        run_cmd(
            f"gsutil ls {GCS_BUCKET}/processed/training_features/",
            "Verify training features output"
        )
        with self.output().open("w") as f:
            f.write("ok")


class BuildEventWindow(luigi.Task):
    """
    For each past host, create indicator values from t-6 to t+6
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
        run_cmd(
            f"gsutil ls {GCS_BUCKET}/event_window/fact_host_event_window/",
            "Verify event window output"
        )
        with self.output().open("w") as f:
            f.write("ok")


# ===========================================================================
# Local ML task
# ===========================================================================

class TrainAndPredict(luigi.Task):
    """
    Train model on host country profiles (33 rows),
    then score all ~208 countries. Outputs predictions to GCS.
    Runs locally (not on Dataproc).
    """
    def requires(self):
        return [BuildTrainingFeatures()]

    def output(self):
        return luigi.LocalTarget("/tmp/luigi_predictions.flag")

    def run(self):
        import pandas as pd
        import numpy as np
        from sklearn.svm import OneClassSVM
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import pyarrow.parquet as pq
        import gcsfs

        fs = gcsfs.GCSFileSystem(project=GCP_PROJECT)

        # ---- Load raw data ----
        logger.info("Loading data from GCS...")
        wdi_df = pq.ParquetDataset("msba405-team-1-data/raw/wdi/", filesystem=fs).read().to_pandas()
        imf_df = pq.ParquetDataset("msba405-team-1-data/raw/imf/", filesystem=fs).read().to_pandas()
        hosts_df = pd.read_csv(f"gs://{GCS_BUCKET.replace('gs://', '')}/raw/fifa_wc_hosts.csv")

        econ_data = pd.concat([wdi_df, imf_df], ignore_index=True)

        # ---- Build candidate matrix (latest year snapshot) ----
        latest_year = int(econ_data["year"].max())
        logger.info(f"Building candidate matrix for year {latest_year}...")
        cand_df = econ_data[econ_data["year"] == latest_year].pivot(
            index="iso3", columns="indicator_code", values="value"
        )

        # ---- Build host training profiles (snapshot at hosting year) ----
        logger.info("Building host training profiles...")
        host_profiles = []
        for _, row in hosts_df.iterrows():
            snap = econ_data[
                (econ_data["iso3"] == row["iso3"]) & (econ_data["year"] == row["year"])
            ]
            if not snap.empty:
                p = snap.pivot(index="iso3", columns="indicator_code", values="value")
                host_profiles.append(p)

        train_full = pd.concat(host_profiles)
        common = train_full.columns.intersection(cand_df.columns)
        logger.info(f"  {len(train_full)} host profiles, {len(common)} common indicators")

        X_train_raw = train_full[common]
        X_cand_raw = cand_df[common]

        # ---- Log-normalization (manages scale differences) ----
        X_train_log = np.sign(X_train_raw) * np.log1p(np.abs(X_train_raw))
        X_cand_log = np.sign(X_cand_raw) * np.log1p(np.abs(X_cand_raw))

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_log))
        X_cand_scaled = scaler.transform(imputer.transform(X_cand_log))

        # ---- PCA with orientation calibration ----
        logger.info("Running PCA...")
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_cand_pca = pca.transform(X_cand_scaled)

        # Calibrate orientation: ensure high-capacity countries are on the positive side
        if "USA" in cand_df.index and "ZWE" in cand_df.index:
            usa_pos = X_cand_pca[cand_df.index.get_loc("USA"), 0]
            zwe_pos = X_cand_pca[cand_df.index.get_loc("ZWE"), 0]
            if usa_pos < zwe_pos:
                X_train_pca[:, 0] *= -1
                X_cand_pca[:, 0] *= -1
                logger.info("  Calibrated PCA orientation.")

        # ---- Train One-Class SVM ----
        logger.info("Training One-Class SVM...")
        ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
        ocsvm.fit(X_train_scaled)

        # ---- Score all countries (viability index) ----
        logger.info("Computing viability scores...")
        host_center = X_train_pca.mean(axis=0)

        # Compute host baseline scores for normalization
        host_raw_scores = []
        for h_pca in X_train_pca:
            v = h_pca - host_center
            host_raw_scores.append((v[0] * 1.5) + (v[1] * 1.0))
        host_raw_scores = np.array(host_raw_scores)

        # Score each candidate country
        results_rows = []
        for i, iso3 in enumerate(cand_df.index):
            vec_scaled = X_cand_scaled[i].reshape(1, -1)
            vec_pca = X_cand_pca[i]

            # Power score: Scale (PC1) + Stability (PC2)
            vector = vec_pca - host_center
            power_score = (vector[0] * 1.5) + (vector[1] * 1.0)

            # Viability index: normalize against host distribution (avg host = 50)
            viability_idx = float(np.interp(
                power_score,
                [host_raw_scores.min(), np.mean(host_raw_scores), host_raw_scores.max()],
                [20, 50, 95]
            ))

            # SVM inlier/outlier check
            is_inlier = int(ocsvm.predict(vec_scaled)[0])

            # Data completeness
            completeness = float(cand_df.loc[iso3, common].notna().mean())

            results_rows.append({
                "iso3": iso3,
                "anomaly_label": is_inlier,
                "data_completeness": round(completeness, 4),
                "hosting_readiness_score": round(viability_idx, 2)
            })

        results = pd.DataFrame(results_rows)
        logger.info(f"  Scored {len(results)} countries")

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
    Load all three fact tables into Snowflake from GCS via external stage.
    Implements atomic rollback: if any table fails to load, all tables
    created during this run are dropped to avoid inconsistent state.

    Tables loaded into FACTS schema:
      1. FACT_HOST_EVENT_WINDOW
      2. COUNTRY_SCORES
      3. TRAINING_FEATURES
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
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
        )
        cur = conn.cursor()

        # Use the FACTS schema for all table operations
        cur.execute(f"USE SCHEMA {FACTS_SCHEMA}")

        # Track which tables were successfully created/loaded in this run
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
                FROM @ANALYTICS.GCS_STAGE/event_window/fact_host_event_window/
                FILE_FORMAT = (FORMAT_NAME = ANALYTICS.PARQUET_FF)
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
                FROM @ANALYTICS.GCS_STAGE/processed/predictions/
                FILE_FORMAT = (FORMAT_NAME = ANALYTICS.PARQUET_FF)
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
            cur.execute("""
                CREATE TABLE TRAINING_FEATURES
                USING TEMPLATE (
                    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
                    FROM TABLE(
                        INFER_SCHEMA(
                            LOCATION => '@ANALYTICS.GCS_STAGE/processed/training_features/',
                            FILE_FORMAT => 'ANALYTICS.PARQUET_FF'
                        )
                    )
                );
            """)
            cur.execute("""
                COPY INTO TRAINING_FEATURES
                FROM @ANALYTICS.GCS_STAGE/processed/training_features/
                FILE_FORMAT = (FORMAT_NAME = ANALYTICS.PARQUET_FF)
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
                ON_ERROR = 'CONTINUE';
            """)
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
# Dimension tables
# ===========================================================================

class LoadDimensions(luigi.Task):
    """
    Load dimension tables into Snowflake DIMENSIONS schema.
    Uses World Bank API for country metadata and indicator names.
    
    Tables loaded:
      1. DIM_COUNTRY (iso3, country_name, region, income_group)
      2. DIM_INDICATOR (indicator_code, indicator_name, source)
    """
    def requires(self):
        return [LoadSnowflake()]

    def output(self):
        return luigi.LocalTarget("/tmp/luigi_dimensions.flag")

    def run(self):
        import snowflake.connector
        import pandas as pd

        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            database=SNOWFLAKE_DATABASE,
            warehouse=SNOWFLAKE_WAREHOUSE,
            role=SNOWFLAKE_ROLE,
        )
        cur = conn.cursor()
        cur.execute(f"USE SCHEMA {DIMENSIONS_SCHEMA}")

        try:
            # ---- DIM_INDICATOR ----
            logger.info("Building DIM_INDICATOR from WDI API...")
            import wbgapi as wb

            wdi_series = wb.series.info()
            wdi_map = {s['id']: s['value'] for s in wdi_series.items}

            imf_map = {
                "NGDP_RPCH": "Real GDP Growth (%)",
                "PCPIPCH": "Inflation Rate (%)",
                "LUR": "Unemployment Rate (%)",
                "BCA": "Current Account Balance (USD bn)",
                "GGXWDG_NGDP": "Government Debt (% of GDP)",
                "BCA_GDP": "Current Account Balance (% of GDP)",
                "AI_PI": "Price Index",
            }

            # Get indicator codes from fact table
            cur.execute(f"SELECT DISTINCT indicator_code FROM {FACTS_SCHEMA}.FACT_HOST_EVENT_WINDOW ORDER BY indicator_code")
            all_codes = [row[0] for row in cur.fetchall()]

            cur.execute("DROP TABLE IF EXISTS DIM_INDICATOR")
            cur.execute("""
                CREATE TABLE DIM_INDICATOR (
                    indicator_code VARCHAR,
                    indicator_name VARCHAR,
                    source         VARCHAR
                )
            """)

            for code in all_codes:
                if code in wdi_map:
                    name, source = wdi_map[code], "WDI"
                elif code in imf_map:
                    name, source = imf_map[code], "IMF"
                else:
                    name, source = code, "Unknown"
                # Escape single quotes in indicator names
                name = name.replace("'", "''")
                cur.execute(f"INSERT INTO DIM_INDICATOR VALUES ('{code}', '{name}', '{source}')")

            logger.info(f"  DIM_INDICATOR: {len(all_codes)} rows loaded")

            # ---- DIM_COUNTRY ----
            logger.info("Building DIM_COUNTRY from WDI API...")
            country_info = wb.economy.info()

            cur.execute("DROP TABLE IF EXISTS DIM_COUNTRY")
            cur.execute("""
                CREATE TABLE DIM_COUNTRY (
                    iso3          VARCHAR,
                    country_name  VARCHAR,
                    region        VARCHAR,
                    income_group  VARCHAR
                )
            """)

            count = 0
            for c in country_info.items:
                iso3 = c['id']
                name = c['value'].replace("'", "''")
                region = c.get('region', '') or ''
                income = c.get('incomeLevel', '') or ''
                cur.execute(f"INSERT INTO DIM_COUNTRY VALUES ('{iso3}', '{name}', '{region}', '{income}')")
                count += 1

            logger.info(f"  DIM_COUNTRY: {count} rows loaded")
            logger.info("All dimension tables loaded successfully.")

            with self.output().open("w") as f:
                f.write("ok")

        except Exception as e:
            logger.error(f"Dimension table load FAILED: {e}")
            cur.execute("DROP TABLE IF EXISTS DIM_INDICATOR")
            cur.execute("DROP TABLE IF EXISTS DIM_COUNTRY")
            raise
        finally:
            cur.close()
            conn.close()


# ===========================================================================
# Top-level task
# ===========================================================================

class RunAll(luigi.WrapperTask):
    """Run the full pipeline end-to-end."""
    def requires(self):
        return [LoadDimensions()]


if __name__ == "__main__":
    luigi.run()
