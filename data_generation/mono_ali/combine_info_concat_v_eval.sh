#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J combine_info_eval
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --output=<BASE_DIR>/<USERID>/logs/mono_ali/combine_info_eval_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/logs/mono_ali/combine_info_eval_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=10:00:00

set -e

# =========================
# Arguments
# =========================
lang=$1      # e.g., es, fr, de
split=$2     # e.g., train/dev/test

# =========================
# Dataset setup
# =========================
DATASET=<DATASET>   # e.g., cvss or covost_v2
method="concat"

echo "Processing language: $lang"
echo "Processing split: $split"
echo "Processing dataset: $DATASET"

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>
root="${BASE_DIR}/${USERID}"

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

LANG_CODE=$(map_lang "$lang")
TGT_LANG_CODE="en-US"

echo "Language code: $LANG_CODE"
echo "Target language code: $TGT_LANG_CODE"

# =========================
# Input/output setup
# =========================
if [[ "$DATASET" == "cvss" ]]; then

    src_manifest_dir="${root}/data/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/nemo_prepared_${split}/sharded_manifests"
    src_tar_dir="${root}/data/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/nemo_prepared_${split}"

elif [[ "$DATASET" == "covost_v2" ]]; then

    src_manifest_dir="${root}/data/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/sharded_manifests"
    src_tar_dir="${root}/data/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}"

else
    echo "❌ Invalid DATASET: $DATASET"
    exit 1
fi

results_dir="${root}/lhotse_data_eval/${DATASET}/${LANG_CODE}_${TGT_LANG_CODE}/${split}"

mkdir -p "${results_dir}"
mkdir -p "${root}/logs/mono_ali"

# =========================
# Environment setup
# =========================
source ~/.bashrc
conda activate <CONDA_ENV_NAME>

echo "Using Python at: $(which python)"
echo "Source manifest dir: ${src_manifest_dir}"
echo "Source tar dir: ${src_tar_dir}"
echo "Results dir: ${results_dir}"

# =========================
# Collect manifests
# =========================
shuffled_manifests=$(ls "${src_manifest_dir}"/manifest_*.jsonl | shuf)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

echo "Found ${num_manifests} manifests."

# =========================
# Build evaluation cuts
# =========================
for i in $(seq 0 $((num_manifests - 1))); do

    src_manifest_file="${manifest_array[$i]}"
    manifest_number=$(basename "$src_manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')

    src_tar="${src_tar_dir}/audio_${manifest_number}.tar"

    if [[ -f "${results_dir}/lock_${manifest_number}.done" ]]; then
        echo "🔒 Lock file found for manifest ${manifest_number} – already processed."
    else
        echo "🚀 Running combine step for manifest: ${manifest_number}"

        if python combine_info_eval.py \
            --src_tar "$src_tar" \
            --src_manifest "$src_manifest_file" \
            --output_dir "$results_dir" \
            --method "$method"; then

            touch "${results_dir}/lock_${manifest_number}.done"
        else
            echo "❌ combine_info_eval.py failed for ${manifest_number}"
            exit 1
        fi
    fi
done

# =========================
# Post-processing check
# =========================
total_manifests=$(find "${src_manifest_dir}" -name "manifest_*.jsonl" | wc -l)
total_done=$(find "${results_dir}" -name "lock_*.done" | wc -l)

if [ "$total_manifests" -eq "$total_done" ]; then
    echo "✅ All manifests have been processed!"
else
    echo "⚠️ Not all manifests have been processed: ${total_done} of ${total_manifests}"
fi