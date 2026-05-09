#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J prep_align
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=0-9
#SBATCH --output=<BASE_DIR>/<USERID>/results/text_align/data_prep/logs/alignments_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/results/text_align/data_prep/logs/alignments_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=5:00:00

set -e

# =========================
# Arguments
# =========================
LANG=$1        # e.g., es, fr, de
TGT_LANG=$2    # e.g., en
subset=$3      # e.g., train/dev/test

# =========================
# Dataset setup
# =========================
DATASET=<DATASET>   # e.g., covost_v2 or cvss
sharded_manifests=sharded_manifests

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>

DATA_ROOT="${BASE_DIR}/${USERID}/data"
RESULTS_ROOT="${BASE_DIR}/${USERID}/results"

# =========================
# Language mapping
# =========================
map_lang() {
  case "$1" in
    fr) echo "fr-FR" ;;
    en) echo "en-US" ;;
    de) echo "de-DE" ;;
    es) echo "es-US" ;;
    *)  echo "$1" ;;
  esac
}

LANG_CODE=$(map_lang "$LANG")
TGT_LANG_CODE=$(map_lang "$TGT_LANG")

echo "Processing ${DATASET} ${LANG_CODE} ${TGT_LANG_CODE} ${subset}"

# =========================
# Input/output setup
# =========================
if [[ "$DATASET" == "covost_v2" ]]; then

    INPUT_BASE="${DATA_ROOT}/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/${sharded_manifests}"
    OUTPUT_DIR="${RESULTS_ROOT}/text_align/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"

elif [[ "$DATASET" == "cvss" ]]; then

    INPUT_BASE="${DATA_ROOT}/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/nemo_prepared_${subset}/${sharded_manifests}"
    OUTPUT_DIR="${RESULTS_ROOT}/text_align/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/${subset}"

else
    echo "❌ Unsupported DATASET: $DATASET"
    exit 1
fi

stage=0
stop_stage=0

# =========================
# Create output directories
# =========================
mkdir -p "$OUTPUT_DIR"
mkdir -p "${RESULTS_ROOT}/text_align/data_prep/logs"

# =========================
# Environment setup
# =========================
source ~/.bashrc
conda activate <CONDA_ENV_NAME>

echo "Using Python at: $(which python)"
echo "Input base: $INPUT_BASE"
echo "Output dir: $OUTPUT_DIR"

# =========================
# Collect manifests
# =========================
shuffled_manifests=$(ls "$INPUT_BASE"/manifest_*.jsonl | shuf)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

echo "Found ${num_manifests} manifests."

# =========================
# Process manifests
# =========================
for i in $(seq 0 $(( num_manifests - 1 ))); do
    manifest_file="${manifest_array[$i]}"
    manifest_number=$(basename "$manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')

    echo "Manifest number: $manifest_number"
    echo "Manifest file: $manifest_file"
    echo "Output directory: $OUTPUT_DIR"

    bash process_all_manifests.sh "$manifest_file" "$OUTPUT_DIR" "$stage" "$stop_stage"
done

# =========================
# Post-processing check
# =========================
total_manifests=$(find "${INPUT_BASE}/" -type f -name "manifest_*.jsonl" | wc -l)
total_parallel_done=$(find "${OUTPUT_DIR}/" -name "*.parallel" | wc -l)

if [ "$total_manifests" -eq "$total_parallel_done" ]; then
    echo "✅ All manifest files have been processed!"
else
    echo "⚠️ Not all manifest files have been processed: ${total_parallel_done} of ${total_manifests}"
fi