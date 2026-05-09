#!/bin/bash

# Directories
base_dir="/lustre/fsw/portfolios/adlr/projects/adlr_audio_speech/datasets/LibriTTS/24khz"
output_dir="/lustre/fsw/portfolios/adlr/users/sanggill/datasets/LibriTTS-dev-benchmark"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Function to copy files listed in a text file
copy_files() {
    local txt_file=$1
    local prefix=$2

    while IFS="|" read -r filepath _; do
        src="$base_dir/$filepath.wav"
        dest="$output_dir/$(basename $filepath).wav"
        cp "$src" "$dest"
    done < "$txt_file"
}

# Copy files from dev-clean.txt
copy_files "LibriTTS/dev-clean.txt" "dev-clean"

# Copy files from dev-other.txt (if needed)
copy_files "LibriTTS/dev-other.txt" "dev-other"

echo "Copying completed."
