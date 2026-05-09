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

# =========================
# Model checkpoint setup
# =========================
if [[ "$LANG" == "es" ]]; then
    echo "This is a Spanish $LANG job."

    model_dir="${RESULTS_ROOT}/text_align/asr_es-US_v1.0en/mbert_softmax_40k_plus/checkpoint-4200"

elif [[ "$LANG" == "fr" ]]; then
    echo "This is a French $LANG job."

    model_dir="${RESULTS_ROOT}/text_align/asr_fr-FR_v2.1en/mbert_softmax_40k_plus/checkpoint-2400"

elif [[ "$LANG" == "de" ]]; then
    echo "This is a German $LANG job."

    model_dir="${RESULTS_ROOT}/text_align/asr_de-DE_v1.0en/mbert_softmax_40k_plus/checkpoint-800"

else
    echo "Unknown language job: $LANG"
    exit 1
fi

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

stage=2
stop_stage=2

# =========================
# Create output directory
# =========================
mkdir -p "$OUTPUT_DIR"
mkdir -p "${RESULTS_ROOT}/text_align/logs"

echo "Input base: $INPUT_BASE"
echo "Output dir: $OUTPUT_DIR"
echo "Model dir: $model_dir"

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
    echo "✅ All manifest files have been processed!"
else
    echo "⚠️ Not all manifest files have been processed: ${total_parallel_done} of ${total_manifests}"
fi