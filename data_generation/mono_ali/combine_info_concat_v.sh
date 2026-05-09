#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J combine_info
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=0-9
#SBATCH --output=<BASE_DIR>/<USERID>/logs/mono_ali/combine_info_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/logs/mono_ali/combine_info_%A_%a.err
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --time=10:00:00

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
root="${BASE_DIR}/${USERID}"

# =========================
# Language/data setup
# =========================
if [[ "$LANG" == "es" ]]; then
    echo "This is a Spanish $LANG job."

    DATASET_NAME=asr_es-US_v1.0
    # DATASET_NAME=speech_v1.0

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

    src_manifest_dir="${root}/source_data/${DATASET_NAME}/${lang_dir}/${sharded_manifests}"
    src_tar_dir="${root}/source_data/${DATASET_NAME}/${lang_dir}"
    src_ctm_dir="${root}/results/mfa/${DATASET_NAME}${lang_dir}"

    txt_alignment_dir="${root}/results/text_align/${DATASET_NAME}${lang_dir}"

    tgt_manifest_dir="${root}/results/tts_outputs/${DATASET_NAME}${lang_dir}/${sharded_manifests}"
    tgt_tar_dir="${root}/results/tts_outputs/${DATASET_NAME}${lang_dir}"
    tgt_ctm_dir="${root}/results/mfa/tts_outputs/${DATASET_NAME}${lang_dir}"

    results_dir="${root}/lhotse_data_concat/${DATASET_NAME}${lang_dir}"

else
    bucket=4

    src_manifest_dir="${root}/source_data/${DATASET_NAME}/bucket${bucket}/${sharded_manifests}"
    src_tar_dir="${root}/source_data/${DATASET_NAME}/bucket${bucket}"
    src_ctm_dir="${root}/results/mfa/${DATASET_NAME}${TGT_LANG}/bucket${bucket}"

    txt_alignment_dir="${root}/results/text_align/${DATASET_NAME}${TGT_LANG}/bucket${bucket}"

    tgt_manifest_dir="${root}/results/tts_outputs/${DATASET_NAME}${TGT_LANG}/bucket${bucket}/${sharded_manifests}"
    tgt_tar_dir="${root}/results/tts_outputs/${DATASET_NAME}${TGT_LANG}/bucket${bucket}"
    tgt_ctm_dir="${root}/results/mfa/tts_outputs/${DATASET_NAME}${TGT_LANG}/bucket${bucket}"

    results_dir="${root}/lhotse_data_concat/${DATASET_NAME}/bucket${bucket}"
fi

mkdir -p "${results_dir}"
mkdir -p "${root}/logs/mono_ali"

# =========================
# Environment setup
# =========================
source ~/.bashrc
conda activate <CONDA_ENV_NAME>

echo "Using Python at: $(which python)"
echo "Dataset name: ${DATASET_NAME}"
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
    src_tar="${src_tar_dir}/audio_${manifest_number}.tar"

    tgt_manifest_file="${tgt_manifest_dir}/manifest_${manifest_number}.jsonl"
    tgt_tar="${tgt_tar_dir}/audio_${manifest_number}.tar"
    tgt_ctm="${tgt_ctm_dir}/${manifest_number}/ctm/words.combined.ctm"

    manifest_number_padded=$(printf "%06d" "$manifest_number")

    echo "Processing manifest number: ${manifest_number} (padded: ${manifest_number_padded})"

    if [[ -f "${results_dir}/cuts.${manifest_number_padded}.jsonl.gz" ]]; then
        echo "🔒 Output file found for manifest ${manifest_number} – already processed."
    else
        touch "${results_dir}/lock_${manifest_number}.lock"

        echo "🚀 Running combine step for manifest: ${manifest_number}"

        if python combine_info_concat_v.py \
            --src_tar "$src_tar" \
            --tgt_tar "$tgt_tar" \
            --src_manifest "$src_manifest_file" \
            --txt_alignment_path "$txt_alignment_dir" \
            --tgt_manifest "$tgt_manifest_file" \
            --tgt_ctm "$tgt_ctm" \
            --src_ctm "$src_ctm" \
            --output_dir "$results_dir" \
            --method alignments \
            --chunk_size_ms 2000; then

            touch "${results_dir}/lock_${manifest_number}.done"
        else
            echo "❌ combine_info_concat_v.py failed for ${manifest_number}"
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