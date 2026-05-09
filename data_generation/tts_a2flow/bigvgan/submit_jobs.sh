#!/bin/bash

# Default values
default_filelist="experiments/experiments_ae.txt"
default_image="/lustre/fsw/portfolios/adlr/users/sanggill/docker/bigvgan:24.02-py3-v2.sqsh"

# Assigning command line arguments with fallback defaults
FILELIST=${1:-$default_filelist}
IMAGE=${2:-$default_image}

# Echoing the filelist and image path, indicating whether they were provided or defaulted
if [ "$FILELIST" == "$default_filelist" ]; then
    echo "No filelist explicitly provided, using default: $FILELIST"
else
    echo "Using provided filelist: $FILELIST"
fi

if [ "$IMAGE" == "$default_image" ]; then
    echo "No image path explicitly provided, using default: $IMAGE"
else
    echo "Using provided image path: $IMAGE"
fi

# Execute the Python script with the provided or default values
python3 submit_jobs.py -f $FILELIST -i $IMAGE

# Remove core dump from failed jobs periodically
rm -f core.*