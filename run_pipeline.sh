#!/bin/bash
# =============================================================================
# run_pipeline.sh — Single-command pipeline execution
# MSBA 405 — Team 1 — "Should We Bid?"
#
# This script creates a Dataproc cluster, runs the full Luigi pipeline,
# and deletes the cluster afterward. If the pipeline fails, the cluster
# is still deleted to avoid wasting credits.
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - GCP project set (gcloud config set project msba405-team-1)
#   - Python packages installed (pip install luigi snowflake-connector-python gcsfs pyarrow pandas numpy scikit-learn)
#   - Snowflake credentials set as environment variables:
#       export SNOWFLAKE_USER="your_username"
#       export SNOWFLAKE_PASSWORD="your_password"
#   - Raw data present in GCS bucket (see README for data download steps)
#
# Usage:
#   bash run_pipeline.sh
# =============================================================================

set -e

PROJECT="msba405-team-1"
REGION="us-central1"
ZONE="us-central1-b"
CLUSTER="msba405-prototype"

# ---- Validate environment ----
echo "============================================"
echo "  FIFA World Cup Pipeline — Full Run"
echo "============================================"
echo ""

if [ -z "$SNOWFLAKE_USER" ] || [ -z "$SNOWFLAKE_PASSWORD" ]; then
    echo "ERROR: Snowflake credentials not set."
    echo "  export SNOWFLAKE_USER=\"your_username\""
    echo "  export SNOWFLAKE_PASSWORD=\"your_password\""
    exit 1
fi

echo "[1/5] Checking GCP authentication..."
gcloud config set project $PROJECT --quiet
echo "  Project: $PROJECT"
echo ""

# ---- Check raw data exists ----
echo "[2/5] Verifying raw data in GCS..."
gsutil ls gs://msba405-team-1-data/raw/wdi/ > /dev/null 2>&1 || { echo "ERROR: WDI data not found in GCS. See README for download steps."; exit 1; }
gsutil ls gs://msba405-team-1-data/raw/imf/ > /dev/null 2>&1 || { echo "ERROR: IMF data not found in GCS. See README for download steps."; exit 1; }
gsutil ls gs://msba405-team-1-data/raw/fifa_wc_hosts.csv > /dev/null 2>&1 || { echo "ERROR: FIFA hosts CSV not found in GCS. See README for download steps."; exit 1; }
gsutil ls gs://msba405-team-1-data/scripts/build_training_features.py > /dev/null 2>&1 || { echo "ERROR: PySpark scripts not found in GCS. See README for upload steps."; exit 1; }
gsutil ls gs://msba405-team-1-data/scripts/build_event_window.py > /dev/null 2>&1 || { echo "ERROR: PySpark scripts not found in GCS. See README for upload steps."; exit 1; }
echo "  All raw data and scripts verified."
echo ""

# ---- Create Dataproc cluster ----
echo "[3/5] Creating Dataproc cluster '$CLUSTER'..."
echo "  (This takes 2-3 minutes)"
gcloud dataproc clusters create $CLUSTER \
    --region=$REGION \
    --zone=$ZONE \
    --master-machine-type=n2-standard-2 \
    --worker-machine-type=n2-standard-2 \
    --num-workers=2 \
    --master-boot-disk-size=100GB \
    --worker-boot-disk-size=100GB \
    --image-version=2.2-debian12 \
    --enable-component-gateway \
    --max-idle=2h \
    --quiet
echo "  Cluster created successfully."
echo ""

# ---- Run Luigi pipeline ----
echo "[4/5] Running Luigi pipeline..."
echo ""

# Clear any stale flag files from previous runs
rm -f /tmp/luigi_*.flag

# Run the pipeline. If it fails, we still want to delete the cluster,
# so we capture the exit code instead of letting set -e kill the script.
set +e
python pipeline.py RunAll --local-scheduler
PIPELINE_EXIT=$?
set -e

echo ""

# ---- Delete Dataproc cluster ----
echo "[5/5] Deleting Dataproc cluster '$CLUSTER'..."
gcloud dataproc clusters delete $CLUSTER --region=$REGION --quiet
echo "  Cluster deleted."
echo ""

# ---- Summary ----
echo "============================================"
if [ $PIPELINE_EXIT -eq 0 ]; then
    echo "  Pipeline completed successfully!"
    echo "  Snowflake tables loaded in FIFA_WC.ANALYTICS"
    echo "  Tableau dashboard ready to connect."
else
    echo "  Pipeline FAILED (exit code $PIPELINE_EXIT)"
    echo "  Check logs above for errors."
    echo "  Snowflake tables were rolled back (if applicable)."
    echo "  Cluster was still deleted to save credits."
fi
echo "============================================"

exit $PIPELINE_EXIT
