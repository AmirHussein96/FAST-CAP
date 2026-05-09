#!/bin/bash
# prepare sqsh with below command first
# enroot import -o bigvgan:24.02-py3-v2.sqsh docker://gitlab-master.nvidia.com/sanggill/bigvgan:24.02-py3-v2
# Set NUM_GPUS to the first argument, or 1 if no argument is provided
# Check if the first argument is provided
if [ -z "$1" ]; then
    echo "Warning: No GPU count provided. Defaulting to 1 GPU."
    NUM_GPUS=1
else
    NUM_GPUS=$1
fi

# Check for image argument
if [ -z "$2" ]; then
    echo "Warning: No image provided. Falling back to default image."
    IMAGE="/lustre/fsw/portfolios/adlr/users/sanggill/docker/bigvgan:24.02-py3-v2.sqsh"
else
    IMAGE=$2
fi

echo "Using --image $IMAGE with --gpu $NUM_GPUS"
submit_job -i -n bigvgan --email_mode never --duration 2 --gpu $NUM_GPUS --image $IMAGE
# pulling image every time is slower
#--image gitlab-master.nvidia.com/sanggill/bigvgan:24.02-py3-v2