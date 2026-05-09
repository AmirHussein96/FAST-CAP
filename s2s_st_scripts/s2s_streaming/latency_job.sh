#!/bin/bash
##SBATCH -A llmservice_nemo_mlops
#SBATCH -A edgeai_riva_rivamlops
#SBATCH -J "latency"
#SBATCH --partition=polar,polar3,polar4,interactive
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 4:00:00
#SBATCH --time-min=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=40GB

set -x

LANG=$1
TGT_LANG=$2
MODE=$3

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-nemo}"

# Replace this with the actual path to your latency folder.
LATENCY_DIR="${LATENCY_DIR:-/path/to/s2s_latency}"

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
echo "Mode: $MODE"

RESULTS_DIR="${LATENCY_DIR}/${LANG_CODE}_${TGT_LANG_CODE}"
mkdir -p "${RESULTS_DIR}"

if [[ "$LANG" == "es" ]]; then 
    if [[ "$MODE" == "fixed_chunk" ]]; then 
        model="<model-path>.ckpt"
        model_config="<model-config-path>.yaml"
        output_dir="<output-dir>"
    elif [[ "$MODE" == "target_aligned" ]]; then
        model="<model-path>.ckpt"
        model_config="<model-config-path>.yaml"
        output_dir="<output-dir>"
    fi

elif [[ "$LANG" == "de" ]]; then 
    if [[ "$MODE" == "fixed_chunk" ]]; then 
        model="<model-path>.ckpt"
        model_config="<model-config-path>.yaml"
        output_dir="<output-dir>"
    elif [[ "$MODE" == "target_aligned" ]]; then
        model="<model-path>.ckpt"
        model_config="<model-config-path>.yaml"
        output_dir="<output-dir>"
    fi

elif [[ "$LANG" == "fr" ]]; then 
    if [[ "$MODE" == "fixed_chunk" ]]; then 
        model="<model-path>.ckpt"
        model_config="<model-config-path>.yaml"
        output_dir="<output-dir>"
    elif [[ "$MODE" == "target_aligned" ]]; then
        model="<model-path>.ckpt"
        model_config="<model-config-path>.yaml"
        output_dir="<output-dir>"
    fi
fi

read -r -d '' cmd <<EOF
source "$CONDA_SH" \
&& conda activate "$CONDA_ENV" \
&& cd "$LATENCY_DIR" \
&& echo "current working dir: \$(pwd)" \
&& echo "Using Python at: \$(which python)" \
&& echo "Model: $model" \
&& echo "model_config: $model_config" \
&& echo "output dir: $output_dir" \
&& simuleval --agent s2s_st_agent2.py \
    --latency-metrics LAAL MAAL \
    --source-segment-size 80 \
    --source ${LANG_CODE}_${TGT_LANG_CODE}/source.txt \
    --target ${LANG_CODE}_${TGT_LANG_CODE}/target.txt \
    --output ${LANG_CODE}_${TGT_LANG_CODE}/$output_dir \
    --ctm-path ${LANG_CODE}_${TGT_LANG_CODE}/words.combined.ctm \
    --t2t-align-path ${LANG_CODE}_${TGT_LANG_CODE}/ \
    --quality-metrics BLEU \
    --config-path $model_config \
    --model-path $model
EOF

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

srun -o "$OUTFILE" -e "$ERRFILE" bash -c "${cmd}"