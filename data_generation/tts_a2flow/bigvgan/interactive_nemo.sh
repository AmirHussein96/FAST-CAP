#!/bin/bash
# prepare sqsh with below command first
# enroot import -o bigvgan:nemo.sqsh docker://gitlab-master.nvidia.com/sanggill/bigvgan:nemo
# Set NUM_GPUS to the first argument, or 1 if no argument is provided
# Check if the first argument is provided
if [ -z "$1" ]; then
    echo "Warning: No GPU count provided. Defaulting to 1 GPU."
    NUM_GPUS=1
else
    NUM_GPUS=$1
fi
submit_job -i -n bigvgan --email_mode never --duration 2 --gpu $NUM_GPUS \
--image /lustre/fsw/portfolios/adlr/users/sanggill/docker/bigvgan:nemo.sqsh
# pulling image every time is slower
#--image gitlab-master.nvidia.com/sanggill/bigvgan:nemo