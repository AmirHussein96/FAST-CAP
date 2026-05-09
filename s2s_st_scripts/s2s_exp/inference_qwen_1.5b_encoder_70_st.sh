#!/bin/bash
#SBATCH -A llmservice_nemo_mlops
#SBATCH -J "s2s_duplex_hf_st"
#SBATCH -p batch,batch_large,interactive
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --time-min=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --overcommit
#SBATCH --mem=0

set -euo pipefail
set -x

########################################
# User-configurable settings
########################################

# Conda environment
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-nemo}"

# Seed
SEED="${SEED:-42}"

# Repo / code
CODE_DIR="${CODE_DIR:-/path/to/NeMo}"

# Config
CONFIG_PATH="${CONFIG_PATH:-/path/to/configs/train}"
CONFIG_NAME="${CONFIG_NAME:-qwen_1b_st_multiling_offline}"

# Experiment
EXP_SUFFIX="${EXP_SUFFIX:-encoder_70_st_concat_v_mfa4_padded_eval}"
EXP_NAME="${EXP_NAME:-DFW_${CONFIG_NAME}_${SLURM_JOB_NUM_NODES}nodes_${EXP_SUFFIX}}"

PROJECT_NAME="${PROJECT_NAME:-duplex_s2s_st_exp_concat_v_mfa}"

# Output directory
RESULTS_ROOT="${RESULTS_ROOT:-/path/to/s2s/exp}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${EXP_NAME}}"

# Checkpoint to evaluate or resume from
PRETRAINED_S2S_MODEL="${PRETRAINED_S2S_MODEL:-/path/to/checkpoints/step11001.ckpt}"

# Cache directory
CACHE_DIR="${CACHE_DIR:-/path/to/HFCACHE}"

# LLM
PRETRAINED_LLM="${PRETRAINED_LLM:-Qwen/Qwen2.5-1.5B-Instruct}"

mkdir -p "${RESULTS_DIR}"

OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

########################################
# Basic checks
########################################

if [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: Cannot find conda.sh at: ${CONDA_SH}"
    echo "Set CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

if [ ! -d "${CODE_DIR}" ]; then
    echo "ERROR: CODE_DIR does not exist: ${CODE_DIR}"
    exit 1
fi

if [ ! -d "${CONFIG_PATH}" ]; then
    echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}"
    exit 1
fi

if [ ! -f "${PRETRAINED_S2S_MODEL}" ]; then
    echo "ERROR: PRETRAINED_S2S_MODEL does not exist: ${PRETRAINED_S2S_MODEL}"
    exit 1
fi

########################################
# Command
########################################

read -r -d '' CMD <<EOF || true
source "${CONDA_SH}" && \
conda activate "${CONDA_ENV}" && \

echo "Using Python at: \$(which python)" && \
echo "Python version: \$(python --version)" && \
echo "Code dir: ${CODE_DIR}" && \
echo "Config path: ${CONFIG_PATH}" && \
echo "Config name: ${CONFIG_NAME}" && \
echo "Results dir: ${RESULTS_DIR}" && \
echo "Checkpoint: ${PRETRAINED_S2S_MODEL}" && \

chmod -R 777 "${RESULTS_DIR}" || true && \

export PYTHONPATH="${CODE_DIR}:\${PYTHONPATH:-}" && \
export HF_HOME="${CACHE_DIR}" && \
export TORCH_HOME="${CACHE_DIR}" && \
export NEMO_CACHE_DIR="${CACHE_DIR}" && \
export OMP_NUM_THREADS=1 && \
export TOKENIZERS_PARALLELISM=false && \
export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 && \
export HYDRA_FULL_ERROR=1 && \
export TORCH_CUDNN_V8_API_ENABLED=1 && \

python "${CODE_DIR}/examples/speechlm2/s2s_duplex_speech_decoder_st_train.py" \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" \
    ++exp_manager.checkpoint_callback_params.save_top_k=1 \
    exp_manager.name="${EXP_NAME}" \
    ++model.pretrained_s2s_model="${PRETRAINED_S2S_MODEL}" \
    ++model.mask_sequence_loss=True \
    trainer.num_nodes="${SLURM_JOB_NUM_NODES}" \
    exp_manager.explicit_log_dir="${RESULTS_DIR}" \
    data.train_ds.seed="${SEED}" \
    data.validation_ds.seed="${SEED}" \
    ++model.audio_loss_weight=20 \
    ++model.speech_decoder.cond_on_prev_audio_tokens=True \
    ++model.speech_decoder.use_speaker_encoder=True \
    ++model.speech_decoder.cond_on_char_embedding=True \
    ++model.speech_decoder.cond_on_asr_emb=False \
    ++model.speech_decoder.cond_on_llm_latent=False \
    ++model.speech_decoder.cond_on_modality_adapter_emb=False \
    ++model.speech_decoder.cond_on_text_tokens=False \
    ++model.speech_decoder.cfg_scale=2.5 \
    ++model.speech_decoder.kernel_size=3 \
    ++model.speech_decoder.cfg_unconditional_prob=0.2 \
    ++model.custom_codebook_size=2045 \
    ++model.custom_speech_bos_id=2019 \
    ++model.custom_speech_eos_id=2020 \
    ++model.custom_speech_delay_id=2018 \
    model.perception.encoder.att_context_size=[70,0] \
    model.perception.modality_adapter.att_context_size=[70,0] \
    ++model.pretrained_llm="${PRETRAINED_LLM}" \
    ++model.scale_loss_by="non_sil_t" \
    ++model.scale_loss_mask=10 \
    ++model.use_old_noise_aug=False \
    ++model.val_acc_tolerance=480 \
    ++trainer.limit_val_batches=null \
    ++trainer.val_check_interval=1 \
    ++trainer.max_steps=1
EOF

########################################
# Launch regular Slurm job, no container
########################################

srun -o "${OUTFILE}" -e "${ERRFILE}" bash -lc "${CMD}"