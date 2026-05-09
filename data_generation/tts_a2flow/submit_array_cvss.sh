#!/bin/bash
#SBATCH -A <ACCOUNT>
#SBATCH -J tts_inference
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=<BASE_DIR>/<USERID>/logs/tts_inference_%A_%a.out
#SBATCH --error=<BASE_DIR>/<USERID>/logs/tts_inference_%A_%a.err

set -e

# =========================
# Arguments
# =========================
LANG=$1        # e.g., es, en, fr, de
TGT_LANG=$2    # e.g., en
subset=$3      # e.g., train/dev/test

DELAY=$(( SLURM_ARRAY_TASK_ID * 20 ))
echo "Delaying start for $DELAY seconds..."
sleep "$DELAY"

# =========================
# Dataset setup
# =========================
DATASET=<DATASET>   # e.g., cvss or covost_v2
sharded_manifests=sharded_manifests

# =========================
# Generic paths
# =========================
USERID=<USERID>
BASE_DIR=<BASE_DIR>

DATA_ROOT="${BASE_DIR}/${USERID}/data"
RESULTS_ROOT="${BASE_DIR}/${USERID}/results"
LOG_ROOT="${BASE_DIR}/${USERID}/logs"
MODEL_ROOT="${BASE_DIR}/${USERID}/pretrained_models"
WORKSPACE_ROOT="${BASE_DIR}/${USERID}"

DATA_DIR="${DATA_ROOT}/${DATASET}/${LANG}_${TGT_LANG}/nemo_prepared_${subset}"
CTM_DIR="${RESULTS_ROOT}/mfa/${DATASET}/${LANG}_${TGT_LANG}/${subset}"
RESULTS_DIR="${RESULTS_ROOT}/tts_outputs/${DATASET}/${LANG}_${TGT_LANG}/${subset}"

PRETRAINED_MODEL_DIR="${MODEL_ROOT}/tts_model"

TTS_CODE_DIR="${WORKSPACE_ROOT}/<PATH_TO_TTS_PIPELINE>"

mkdir -p "${LOG_ROOT}"
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/${sharded_manifests}"

# =========================
# Environment setup
# =========================
echo "******* STARTING ********"

source ~/.bashrc
conda activate <CONDA_ENV_NAME>

echo "Using Python at: $(which python)"
echo "DATA_DIR: ${DATA_DIR}"
echo "CTM_DIR: ${CTM_DIR}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "PRETRAINED_MODEL_DIR: ${PRETRAINED_MODEL_DIR}"
echo "TTS_CODE_DIR: ${TTS_CODE_DIR}"

# =========================
# Collect manifests
# =========================
shuffled_manifests=$(ls "${DATA_DIR}/${sharded_manifests}"/manifest_*.jsonl | shuf)
manifest_array=($shuffled_manifests)
num_manifests=${#manifest_array[@]}

echo "Found ${num_manifests} manifests."

# =========================
# Run TTS inference
# =========================
for i in $(seq 0 $((num_manifests - 1))); do
    manifest_file="${manifest_array[$i]}"
    manifest_name=$(basename "$manifest_file" .jsonl)

    manifest_number=$(basename "$manifest_file" | sed -E 's/manifest_([0-9]+)\.jsonl/\1/')

    output_dir="${RESULTS_DIR}"

    lockfile="${output_dir}/${sharded_manifests}/${manifest_name}.lock"
    donefile="${output_dir}/${sharded_manifests}/${manifest_name}.done"

    mkdir -p "${output_dir}/${sharded_manifests}"

    output_tar="${output_dir}/audio_${manifest_number}.tar"
    output_manifest="${output_dir}/${sharded_manifests}/manifest_${manifest_number}.jsonl"

    echo "output_tar: ${output_tar}"
    echo "output_manifest: ${output_manifest}"

    if [ -f "$lockfile" ] || [ -f "$donefile" ]; then
        echo "✅ Skipping ${manifest_name} – already exists."
        continue
    else
        echo "🔁 Processing: ${manifest_name}"

        # Manually remove stale .lock files before rerunning a failed job.
        touch "$lockfile"

        cd "$TTS_CODE_DIR"

        python tts_generate_long_resumetar.py \
            --tar "${DATA_DIR}/audio_${manifest_number}.tar" \
            --manifest "${DATA_DIR}/${sharded_manifests}/manifest_${manifest_number}.jsonl" \
            --ctm "${CTM_DIR}/${manifest_number}/ctm/words.combined.ctm" \
            --generator_path "${PRETRAINED_MODEL_DIR}/<GENERATOR_MODEL_DIR>" \
            --dp_path "${PRETRAINED_MODEL_DIR}/<DURATION_MODEL_DIR>" \
            --vocoder_path "${PRETRAINED_MODEL_DIR}/<VOCODER_CHECKPOINT>" \
            --vocoder_config_path "${PRETRAINED_MODEL_DIR}/<VOCODER_CONFIG>" \
            --output_dir "$output_dir" \
            --batch_size 1 \
            --num_workers 1 \
            --max_segment_duration 5.0 \
            --max_pause 1.0 \
            --text_processors configs/<TEXT_PROCESSOR_CONFIG>.json \
            --ckpt_num <CHECKPOINT_NUMBER> \
            --sharded_dir "$sharded_manifests" \
            --adaptive_dur_scale \
            --cleanup-wav

        if [ -f "$output_tar" ] && [ -f "$output_manifest" ]; then
            echo "✅ Done with manifest ${manifest_number}"
            touch "$donefile"
        else
            echo "❌ Failed to process manifest ${manifest_name}"
        fi
    fi
done

# =========================
# Optional post-processing check
# =========================
# total_audio=$(find "${DATA_DIR}/" -type f -name "audio_*.tar" | wc -l)
# total_audio_done=$(find "${RESULTS_DIR}/" -name "audio_*.tar" | wc -l)
#
# if [ "$total_audio" -eq "$total_audio_done" ]; then
#     echo "✅ All audio tar files have been processed!"
# else
#     echo "⚠️ Not all audio tar files have been processed: $total_audio_done of $total_audio"
# fi

