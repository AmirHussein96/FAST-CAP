#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J prep_align
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=0-29
#SBATCH --output=<BASE_DIR>/<USERID>/results/text_align/data_prep/logs/alignments_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/results/text_align/data_prep/logs/alignments_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=5:00:00

set -e

# =========================
# Arguments
# =========================
LANG=$1        # e.g., es, en, fr, de
TGT_LANG=$2    # e.g., en, de

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>

PROJECT_DATA_ROOT="${BASE_DIR}/projects/<PROJECT_NAME>/data/<DATA_SUBDIR>"
RESULTS_ROOT="${BASE_DIR}/${USERID}/results"

# =========================
# Language/data setup
# =========================
if [[ "$LANG" == "es" ]]; then
    echo "This is a Spanish $LANG job."

    # DATASET_NAME=asr_es-US_v1.0
    DATASET_NAME=speech_v1.0

    lang_dir=es-US
    sharded_manifests=sharded_manifests_es-US_en-US

elif [[ "$LANG" == "en" ]]; then
    echo "This is an English $LANG job."

    DATASET_NAME=asr_en-US_v6.0

    if [[ "$TGT_LANG" == "de" ]]; then
        sharded_manifests=sharded_manifests_en-US_de-DE
    fi

elif [[ "$LANG" == "fr" ]]; then
    echo "This is a French $LANG job."

    # DATASET_NAME=asr_fr-FR_v2.1
    DATASET_NAME=speech_v1.0

    lang_dir=fr-FR
    sharded_manifests=sharded_manifests_fr-FR_en-US

elif [[ "$LANG" == "de" ]]; then
    echo "This is a German $LANG job."

    DATASET_NAME=asr_de-DE_v1.0
    # DATASET_NAME=speech_v1.0

    lang_dir=de-DE
    sharded_manifests=sharded_manifests_de-DE_en-US

else
    echo "Unknown language job."
    exit 1
fi

# =========================
# Input/output setup
# =========================
if [[ "$DATASET_NAME" == "speech_v1.0" ]]; then
    echo "Processing speech dataset ${lang_dir}"

    INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET_NAME}/${lang_dir}/${sharded_manifests}"
    OUTPUT_DIR="${RESULTS_ROOT}/text_align/${DATASET_NAME}${lang_dir}"

else
    bucket=1

    INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET_NAME}/bucket${bucket}/${sharded_manifests}"
    OUTPUT_DIR="${RESULTS_ROOT}/text_align/${DATASET_NAME}${TGT_LANG}/bucket${bucket}"
fi

# Optional example for local/untarred data:
# INPUT_BASE="${BASE_DIR}/${USERID}/untarred_data/<DATASET>/<LANG_PAIR>/sharded/${sharded_manifests}"
# OUTPUT_DIR="${RESULTS_ROOT}/text_align/<DATASET>/<LANG_PAIR>"

stage=0
stop_stage=0

# =========================
# Create output directory
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
echo "Dataset name: $DATASET_NAME"
echo "Sharded manifests: $sharded_manifests"

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
    echo "✅ All manifest files have been processed! in bucket ${bucket:-N/A}"
else
    echo "⚠️ Not all manifest files have been processed: ${total_parallel_done} of ${total_manifests} in bucket ${bucket:-N/A}"
fi