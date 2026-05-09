#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J combine_info
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=0-0
#SBATCH --output=<BASE_DIR>/<USERID>/logs/mono_ali/<DATASET>/combine_info_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/logs/mono_ali/<DATASET>/combine_info_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=10:00:00

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
DATASET_NAME=<DATASET>   # e.g., cvss
sharded_manifests=sharded_manifests

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

LANG_CODE=$(map_lang "$LANG")
TGT_LANG_CODE=$(map_lang "$TGT_LANG")

echo "Language code: $LANG_CODE"
echo "Target language code: $TGT_LANG_CODE"
echo "Dataset: $DATASET_NAME"
echo "Subset: $subset"

# =========================
# Input/output setup
# =========================
if [[ "$DATASET_NAME" == "cvss" ]]; then
    echo "Processing ${DATASET_NAME} ${LANG_CODE}_${TGT_LANG_CODE}"

    src_manifest_dir="${root}/data/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}/nemo_prepared_${subset}/${sharded_manifests}"
    src_tar_dir="${root}/data/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}/nemo_prepared_${subset}"

    src_ctm_dir="${root}/results/mfa/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}"
    txt_alignment_dir="${root}/results/text_align/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}"

    tgt_manifest_dir="${root}/results/tts_outputs/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}/${sharded_manifests}"
    tgt_tar_dir="${root}/results/tts_outputs/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}"
    tgt_ctm_dir="${root}/results/mfa/tts_outputs/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}"

    results_dir="${root}/lhotse_data_concat_target_aligned/${DATASET_NAME}/${LANG_CODE}_${TGT_LANG_CODE}"
else
    echo "❌ Unsupported DATASET_NAME: $DATASET_NAME"
    exit 1
fi

mkdir -p "${results_dir}"
mkdir -p "${root}/logs/mono_ali/${DATASET_NAME}"

# =========================
# Environment setup
# =========================
source ~/.bashrc
conda activate <CONDA_ENV_NAME>

echo "Using Python at: $(which python)"
echo "Source manifest dir: ${src_manifest_dir}"
echo "Source tar dir: ${src_tar_dir}"
echo "Source CTM dir: ${src_ctm_dir}"
echo "Text alignment dir: ${txt_alignment_dir}"
echo "Target manifest dir: ${tgt_manifest_dir}"
echo "Target tar dir: ${tgt_tar_dir}"
echo "Target CTM dir: ${tgt_ctm_dir}"
echo "Results dir: ${results_dir}"

# =========================
# Collect manifests
# =========================
shuffled_manifests=$(ls "${src_manifest_dir}"/manifest_*.jsonl | shuf)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

echo "Found ${num_manifests} manifests."

# =========================
# Combine source/target info
# =========================
for i in $(seq 0 $((num_manifests - 1))); do

    src_manifest_file="${manifest_array[$i]}"
    manifest_number=$(basename "$src_manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')

    src_ctm="${src_ctm_dir}/${manifest_number}/ctm/words.combined.ctm"
    src_tar="${src_tar_dir}/audio_${manifest_number}.tar"

    tgt_manifest_file="${tgt_manifest_dir}/manifest_${manifest_number}.jsonl"
    tgt_tar="${tgt_tar_dir}/audio_${manifest_number}.tar"
    tgt_ctm="${tgt_ctm_dir}/${manifest_number}/ctm/words.combined.ctm"

    echo "Processing manifest number: ${manifest_number}"

    if [[ -f "${results_dir}/lock_${manifest_number}.lock" ]] || [[ -f "${results_dir}/lock_${manifest_number}.done" ]]; then
        echo "🔒 Lock file found for manifest ${manifest_number} – already processed."
    else
        touch "${results_dir}/lock_${manifest_number}.lock"

        echo "🚀 Running combine step for manifest: ${manifest_number}"

        if python combine_info_concat_target_aligned.py \
            --src_tar "$src_tar" \
            --tgt_tar "$tgt_tar" \
            --src_manifest "$src_manifest_file" \
            --txt_alignment_path "$txt_alignment_dir" \
            --tgt_manifest "$tgt_manifest_file" \
            --tgt_ctm "$tgt_ctm" \
            --src_ctm "$src_ctm" \
            --output_dir "$results_dir" \
            --method alignments \
            --chunk_min_duration 0.96; then

            touch "${results_dir}/lock_${manifest_number}.done"
        else
            echo "❌ combine_info_concat_target_aligned.py failed for ${manifest_number}"
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