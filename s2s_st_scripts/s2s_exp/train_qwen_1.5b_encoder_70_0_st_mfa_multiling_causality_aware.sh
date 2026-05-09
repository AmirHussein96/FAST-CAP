#!/bin/bash
#SBATCH -A edgeai_riva_rivamlops
#SBATCH -J "s2s_duplex_hf_st"
#SBATCH --partition=polar,polar3,polar4
#SBATCH -N 8
#SBATCH --cpus-per-task=8
#SBATCH -t 4:00:00
#SBATCH --time-min=04:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0

set -euo pipefail
set -x

############################
# User-configurable settings
############################

# Conda
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-nemo_s2s}"

# Project/repo
CODE_DIR="${CODE_DIR:-/path/to/NeMo_S2S}"

# Config
CONFIG_PATH="${CONFIG_PATH:-configs/train}"
CONFIG_NAME="${CONFIG_NAME:-qwen_1b_st_concat_v_mfa2_multiling_target_aligned2}"

# Experiment
EXP_PREFIX="${EXP_PREFIX:-DFW}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX}_${CONFIG_NAME}_${SLURM_JOB_NUM_NODES}nodes_encoder_70_st_concat_v_mfa2_multiling_target_aligned2}"
RESULTS_ROOT="${RESULTS_ROOT:-/path/to/s2s_exp}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${EXP_NAME}}"

PROJECT_NAME="${PROJECT_NAME:-duplex_s2s_st_exp_concat_v_mfa}"
WANDB_ENTITY="${WANDB_ENTITY:-riva-nemo-llm-mlops}"

# Caches
CACHE_DIR="${CACHE_DIR:-/path/to/cache/HFCACHE}"

# Pretrained models
PRETRAINED_AUDIO_CODEC="${PRETRAINED_AUDIO_CODEC:-/path/to/pretrained_models/nano_codec/Low_Frame-rate_Speech_Codec++.nemo}"
PRETRAINED_TTS_FROM_S2S="${PRETRAINED_TTS_FROM_S2S:-/path/to/pretrained_models/magpie_tts/tts-pretraining_qwnen_2.5_81007_steps.ckpt}"
PRETRAINED_ASR="${PRETRAINED_ASR:-/path/to/pretrained_models/asr/mml-en-fr-es-de/en-fr-es-de.nemo}"
PRETRAINED_LLM="${PRETRAINED_LLM:-Qwen/Qwen2.5-1.5B-Instruct}"

# Data / augmentation
NOISE_AUG_PATH="${NOISE_AUG_PATH:-/path/to/data/noise}"

# Seed
SEED="${SEED:-$((SLURM_JOB_ID % 2147483647))}"

# Output logs
mkdir -p "${RESULTS_DIR}"
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

############################
# Environment checks
############################

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set. WandB logging may fail."
fi

if [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: Could not find conda.sh at: ${CONDA_SH}"
    echo "Set CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

if [ ! -d "${CODE_DIR}" ]; then
    echo "ERROR: CODE_DIR does not exist: ${CODE_DIR}"
    exit 1
fi

############################
# Training command
############################

read -r -d '' CMD <<EOF || true
source "${CONDA_SH}" && \
conda activate "${CONDA_ENV}" && \

echo "Using Python at: \$(which python)" && \
echo "Python version: \$(python --version)" && \
echo "Code dir: ${CODE_DIR}" && \
echo "Config path: ${CONFIG_PATH}" && \
echo "Config name: ${CONFIG_NAME}" && \
echo "Results dir: ${RESULTS_DIR}" && \

chmod -R 777 "${RESULTS_DIR}" || true && \

export WANDB_API_KEY="${WANDB_API_KEY:-}" && \
export WANDB_ENTITY="${WANDB_ENTITY}" && \
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
    ++exp_manager.checkpoint_callback_params.save_top_k=3 \
    exp_manager.name="${EXP_NAME}" \
    exp_manager.wandb_logger_kwargs.name="${EXP_NAME}" \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.wandb_logger_kwargs.project="${PROJECT_NAME}" \
    ++exp_manager.wandb_logger_kwargs.entity="${WANDB_ENTITY}" \
    ++exp_manager.wandb_logger_kwargs.resume=true \
    ++model.pretrained_audio_codec="${PRETRAINED_AUDIO_CODEC}" \
    ++model.pretrained_tts_from_s2s="${PRETRAINED_TTS_FROM_S2S}" \
    ++model.pretrained_asr="${PRETRAINED_ASR}" \
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
    ++trainer.limit_val_batches=1 \
    ++trainer.val_check_interval=1000 \
    ++model.scale_loss_by="non_sil_t" \
    ++model.scale_loss_mask=10 \
    ++model.use_old_noise_aug=True \
    ++model.val_acc_tolerance=480 \
    ++model.old_noise_aug_path="${NOISE_AUG_PATH}"
EOF

srun -o "${OUTFILE}" -e "${ERRFILE}" bash -lc "${CMD}"