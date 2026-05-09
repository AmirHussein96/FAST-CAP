#!/bin/bash

################################################################################
# Usage:
#
#   bash run_awesome_align.sh <manifest> <output_dir> <stage> <stop_stage> <model_path> <resume_training>
#
#
# Arguments:
#   manifest          Path to a JSONL manifest file
#   output_dir        Directory to store processed files, checkpoints, outputs
#   stage             Starting stage (0=data prep, 1=train, 2=align)
#   stop_stage        Final stage to execute
#   model_path        HF model name or checkpoint directory
#   resume_training   "true" or "false" — whether to continue training from checkpoint
################################################################################

# Base directories
. ~/.bashrc
conda activate awesome_align
echo "Using Python at: $(which python)"

manifest=$1
OUTPUT_DIR=$2
stage=$3
stop_stage=$4
resume_training=$5
MODEL_NAME_OR_PATH=$6

if [ -z "$manifest" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$stage" ] || [ -z "$stop_stage" ]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: bash $0 <manifest> <output_dir> <stage> <stop_stage> <resume_training> <model_path>"
    exit 1
fi

echo "start_stage: $stage"
echo "stop_stage: $stop_stage"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all manifest files and process them
file_name=$(basename "$manifest" .jsonl)
# --------------------------------------------------------------------------- #
# Stage 0: Data preparation
# --------------------------------------------------------------------------- #
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
echo "stage 0: Data preparation"
    echo "Processing manifest: $file_name"
    if [ ! -f "$OUTPUT_DIR/$file_name.parallel" ]; then
        ./process_manifest.py "$manifest" --output-dir "$OUTPUT_DIR" --clean-tgt
        echo "Done processing manifest: $manifest"
    else
        echo "Skipping manifest: $manifest – already processed."
    fi 
fi

# --------------------------------------------------------------------------- #
# Stage 1: Training
# --------------------------------------------------------------------------- #

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Training"
    if [ "$resume_training" = "true" ]; then
        echo "Resuming training"
        OUTDIR=$OUTPUT_DIR/mbert_softmax_40k_plus
        TRAIN_FILE=$OUTPUT_DIR/train.parallel
        EVAL_FILE=$OUTPUT_DIR/dev.parallel
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 awesome-train \
            --output_dir=$OUTDIR \
            --should_continue \
            --extraction 'softmax' \
            --do_train \
            --cache_data \
            --train_mlm \
            --train_tlm \
            --train_tlm_full \
            --train_so \
            --train_psi \
            --train_data_file=$TRAIN_FILE \
            --per_gpu_train_batch_size 64 \
            --per_gpu_eval_batch_size 64 \
            --gradient_accumulation_steps 2 \
            --num_train_epochs 2 \
            --learning_rate 2e-5 \
            --save_steps 200 \
            --logging_steps 200 \
            --max_steps 40000 \
            --do_eval \
            --overwrite_output_dir \
            --eval_data_file=$EVAL_FILE
    else
        echo "Training from scratch"
        OUTDIR=$OUTPUT_DIR/mbert_softmax_40k_plus
        TRAIN_FILE=$OUTPUT_DIR/train.parallel
        EVAL_FILE=$OUTPUT_DIR/dev.parallel
        MODEL_NAME_OR_PATH=bert-base-multilingual-cased
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 awesome-train \
            --output_dir=$OUTDIR \
            --model_name_or_path=$MODEL_NAME_OR_PATH \
            --extraction 'softmax' \
            --do_train \
            --cache_data \
            --train_mlm \
            --train_tlm \
            --train_tlm_full \
            --train_so \
            --train_psi \
            --train_data_file=$TRAIN_FILE \
            --per_gpu_train_batch_size 64 \
            --gradient_accumulation_steps 2 \
            --per_gpu_eval_batch_size 64 \
            --num_train_epochs 2 \
            --learning_rate 2e-5 \
            --save_steps 200 \
            --logging_steps 200 \
            --max_steps 40000 \
            --do_eval \
            --eval_data_file=$EVAL_FILE
    fi
    
fi

# --------------------------------------------------------------------------- #
# Stage 2: Alignment generation
# --------------------------------------------------------------------------- #

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Generate alignments"
    if [ ! -f "$OUTPUT_DIR/$file_name.awesome-align.out" ]; then
        DATA_FILE=$OUTPUT_DIR/$file_name.parallel
        if [ -z "$MODEL_NAME_OR_PATH" ]; then
            echo "Using default model path."
            MODEL_NAME_OR_PATH=bert-base-multilingual-cased
        fi
        # MODEL_NAME_OR_PATH=bert-base-multilingual-cased
        #MODEL_NAME_OR_PATH=/lustre/fsw/portfolios/edgeai/users/amhussein/results/awesome_align_clean/riva_asr_es-US_v1.0/mbert_softmax_40k_plus/checkpoint-1900 # path to the finetuned model
        OUTPUT_FILE=$OUTPUT_DIR/$file_name.awesome-align.out

        CUDA_VISIBLE_DEVICES=0 awesome-align \
            --output_file=$OUTPUT_FILE \
            --model_name_or_path=$MODEL_NAME_OR_PATH \
            --data_file=$DATA_FILE \
            --extraction 'softmax' \
            --batch_size 2048
        echo "Done aligning: $file_name"

    else
        echo "Skipping alignment: $file_name – already processed."
    fi

fi




# # Optional: Combine all parallel files into one
# echo "Combining all parallel files..."
# cat "$OUTPUT_DIR"/*.parallel > "$OUTPUT_DIR/all.parallel"
# cat "$OUTPUT_DIR"/*.ids > "$OUTPUT_DIR/all.ids"

# echo "Done! Output files are in $OUTPUT_DIR"
# echo "Combined files are:"
# echo "  - $OUTPUT_DIR/all.parallel"
# echo "  - $OUTPUT_DIR/all.ids" 