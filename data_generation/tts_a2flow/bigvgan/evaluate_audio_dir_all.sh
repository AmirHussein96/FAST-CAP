#!/bin/bash

# Path to the experiments directory containing model directories
EXP_DIR="/lustre/fsw/portfolios/adlr/users/sanggill/temp/sample_stable_audio"

# Determine the number of available GPUs using nvidia-smi
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPUS[@]}

# Function to run evaluation
run_evaluation() {
  local data_dir="$1"
  local gpu_id="$2"

  # if [ -f "$data_dir/evaluation_results.json" ]; then
  #   echo "Skipping $data_dir as evaluation_results.json already exists"
  # else
  #   echo "################################################"
  #   echo "Evaluating data in $data_dir on GPU $gpu_id"
  #   CMD="CUDA_VISIBLE_DEVICES=$gpu_id python evaluate_audio_dir.py --audio_dir \"$data_dir\" --skip_mcd_v1"
  #   echo $CMD
  #   eval $CMD &
  #   echo "Started evaluation for $data_dir on GPU $gpu_id"
  #   echo "################################################"
  # fi
  echo "################################################"
  echo "Evaluating data in $data_dir on GPU $gpu_id"
  CMD="CUDA_VISIBLE_DEVICES=$gpu_id python evaluate_audio_dir.py --audio_dir \"$data_dir\" --skip_mcd_v1"
  echo $CMD
  eval $CMD &
  echo "Started evaluation for $data_dir on GPU $gpu_id"
  echo "################################################"
}

# Find all data directories with 'real' and 'generated' subdirectories
data_dirs=()
for model_dir in "$EXP_DIR"/*; do
  if [ -d "$model_dir" ]; then
    for checkpoint_dir in "$model_dir"/*; do
      if [ -d "$checkpoint_dir" ]; then
        for data_dir in "$checkpoint_dir"/*; do
          if [ -d "$data_dir/real" ] && [ -d "$data_dir/generated" ]; then
            data_dirs+=("$data_dir")
          fi
        done
      fi
    done
  fi
done

# Run evaluations in parallel, cycling through available GPUs
for i in "${!data_dirs[@]}"; do
  gpu_id=${GPUS[$((i % NUM_GPUS))]}
  run_evaluation "${data_dirs[$i]}" "$gpu_id"
done

# Wait for all background processes to finish
wait

echo "All evaluations completed."