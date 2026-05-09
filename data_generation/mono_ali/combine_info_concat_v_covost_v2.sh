#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J combine_info
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --array=0-9
#SBATCH --output=<BASE_DIR>/<USERID>/logs/mono_ali/<DATASET>/logs/combine_info_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/logs/mono_ali/<DATASET>/logs/combine_info_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=5:00:00

set -e

# =========================
# Dataset setup
# =========================
DATASET=<DATASET>                  # e.g., covost_v2
dataset_name=<SRC_LANG>_<TGT_LANG> # e.g., es-US_en-US
sharded_manifests=sharded_manifests
chunk_size_ms=2000

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>
root="${BASE_DIR}/${USERID}"

src_manifest_dir="${root}/data/${DATASET}/${dataset_name}/sharded/${sharded_manifests}"
src_tar_dir="${root}/data/${DATASET}/${dataset_name}/sharded"

src_ctm_dir="${root}/results/mfa/${DATASET}/${dataset_name}"
txt_alignment_dir="${root}/results/text_align/${DATASET}/${dataset_name}"

tgt_manifest_dir="${root}/results/tts_outputs/${DATASET}/${dataset_name}/${sharded_manifests}"
tgt_tar_dir="${root}/results/tts_outputs/${DATASET}/${dataset_name}"
tgt_ctm_dir="${root}/results/mfa_tts/${DATASET}/${dataset_name}"

results_dir="${root}/lhotse_data_concat/${DATASET}/${dataset_name}"

mkdir -p "${results_dir}"
mkdir -p "${root}/logs/mono_ali/${DATASET}/logs"

# =========================
# Environment setup
# =========================
source ~/.bashrc
conda activate <CONDA_ENV_NAME>

echo "Using Python at: $(which python)"
echo "Dataset: ${DATASET}"
echo "Dataset name: ${dataset_name}"
echo "Source manifest dir: ${src_manifest_dir}"
echo "Target manifest dir: ${tgt_manifest_dir}"
echo "Results dir: ${results_dir}"

# =========================
# Collect source manifests
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
    src_tar="${src_tar_dir}/audios_${manifest_number}.tar"

    tgt_manifest_file="${tgt_manifest_dir}/manifest_${manifest_number}.jsonl"
    tgt_tar="${tgt_tar_dir}/audios_${manifest_number}.tar"
    tgt_ctm="${tgt_ctm_dir}/${manifest_number}/ctm/words.combined.ctm"

    if [[ -f "${results_dir}/lock_${manifest_number}.done" ]]; then
        echo "🔒 Lock file found for manifest ${manifest_number} – already processed."
    else
        echo "🚀 Running combine step for manifest: ${manifest_number}"

        if python combine_info_concat.py \
            --src_tar "$src_tar" \
            --tgt_tar "$tgt_tar" \
            --src_manifest "$src_manifest_file" \
            --txt_alignment_path "$txt_alignment_dir" \
            --tgt_manifest "$tgt_manifest_file" \
            --tgt_ctm "$tgt_ctm" \
            --src_ctm "$src_ctm" \
            --output_dir "$results_dir" \
            --method alignments \
            --chunk_size_ms "$chunk_size_ms"; then

            touch "${results_dir}/lock_${manifest_number}.done"
        else
            echo "❌ combine_info_concat.py failed for ${manifest_number}"
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