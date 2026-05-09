#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J text_align
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --array=0-9
#SBATCH --output=<BASE_DIR>/<USERID>/results/text_align/logs/alignments_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/results/text_align/logs/alignments_%A_%a.err
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --time=4:00:00

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

    # DATASET=asr_es-US_v1.0
    DATASET=speech_v1.0

    lang_dir=es-US
    sharded_manifests=sharded_manifests_es-US_en-US

    model_dir="${RESULTS_ROOT}/text_align/asr_es-US_v1.0en/mbert_softmax_40k_plus/checkpoint-4200"

elif [[ "$LANG" == "en" ]]; then
    echo "This is an English $LANG job."

    DATASET=asr_en-US_v6.0

    if [[ "$TGT_LANG" == "de" ]]; then
        sharded_manifests=sharded_manifests_en-US_de-DE
    fi

elif [[ "$LANG" == "fr" ]]; then
    echo "This is a French $LANG job."

    # DATASET=asr_fr-FR_v2.1
    DATASET=speech_v1.0

    lang_dir=fr-FR
    sharded_manifests=sharded_manifests_fr-FR_en-US

    model_dir="${RESULTS_ROOT}/text_align/asr_fr-FR_v2.1en/mbert_softmax_40k_plus/checkpoint-2400"

elif [[ "$LANG" == "de" ]]; then
    echo "This is a German $LANG job."

    # DATASET=asr_de-DE_v1.0
    DATASET=speech_v1.0

    lang_dir=de-DE
    sharded_manifests=sharded_manifests_de-DE_en-US

    model_dir="${RESULTS_ROOT}/text_align/asr_de-DE_v1.0en/mbert_softmax_40k_plus/checkpoint-800"

else
    echo "Unknown language job."
    exit 1
fi

# =========================
# Input/output setup
# =========================
if [[ "$DATASET" == "speech_v1.0" ]]; then
    echo "Processing speech dataset ${lang_dir}"

    INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}/${lang_dir}/${sharded_manifests}"
    OUTPUT_DIR="${RESULTS_ROOT}/text_align/${DATASET}${lang_dir}"

else
    bucket=1

    INPUT_BASE="${PROJECT_DATA_ROOT}/${DATASET}/bucket${bucket}/${sharded_manifests}"
    OUTPUT_DIR="${RESULTS_ROOT}/text_align/${DATASET}${TGT_LANG}/bucket${bucket}"
fi

# Optional examples for local/untarred data:
# INPUT_BASE="${BASE_DIR}/${USERID}/untarred_data/<DATASET>/<LANG_PAIR>/sharded/${sharded_manifests}"
# OUTPUT_DIR="${RESULTS_ROOT}/text_align/<DATASET>/<LANG_PAIR>"

stage=2
stop_stage=2

# =========================
# Create output directory
# =========================
mkdir -p "$OUTPUT_DIR"
mkdir -p "${RESULTS_ROOT}/text_align/logs"

echo "Input base: $INPUT_BASE"
echo "Output dir: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Sharded manifests: $sharded_manifests"
echo "Model dir: ${model_dir:-N/A}"

# =========================
# Collect manifests
# =========================
shuffled_manifests=$(ls "$INPUT_BASE"/manifest_*.jsonl | shuf)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

echo "Found ${num_manifests} manifests."

# =========================
# Run alignment stage
# =========================
for i in $(seq 0 $(( num_manifests - 1 ))); do
    manifest_file="${manifest_array[$i]}"
    manifest_number=$(basename "$manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')

    echo "Manifest number: $manifest_number"
    echo "Manifest file: $manifest_file"
    echo "Output directory: $OUTPUT_DIR"

    bash process_all_manifests.sh \
        "$manifest_file" \
        "$OUTPUT_DIR" \
        "$stage" \
        "$stop_stage" \
        false \
        "$model_dir"
done

echo "Script completed successfully"

# =========================
# Post-processing check
# =========================
total_manifests=$(find "${INPUT_BASE}/" -type f -name "manifest_*.jsonl" | wc -l)
total_parallel_done=$(find "${OUTPUT_DIR}/" -name "*.awesome-align.out" | wc -l)

if [ "$total_manifests" -eq "$total_parallel_done" ]; then
    echo "✅ All manifest files have been processed! in bucket ${bucket:-N/A}"
else
    echo "⚠️ Not all manifest files have been processed: ${total_parallel_done} of ${total_manifests} in bucket ${bucket:-N/A}"
fi