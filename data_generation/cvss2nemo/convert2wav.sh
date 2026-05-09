#!/usr/bin/env bash

ROOT="data/cvss"

echo "Scanning: $ROOT"

# Find all mp3 files (excluding already-broken mp3.wav cases)
find "$ROOT" -type f \( -name "*.mp3" -o -name "*.mp3.wav" \) -print0 | while IFS= read -r -d '' mp3; do
    wav="${mp3%.mp3}"
    wav="${wav%.wav}.wav"

    # If wav already exists and is non-empty, skip conversion
    if [[ -f "$wav" ]]; then
        echo "[SKIP] $wav already exists"
        rm -f "$mp3"
        continue
    fi

    echo "[CONVERT] $mp3 -> $wav"
    ffmpeg -nostdin -y -loglevel error \
        -i "$mp3" -ar 16000 -ac 1 "$wav"

    # Verify conversion succeeded
    if [[ -f "$wav" ]]; then
        rm -f "$mp3"
    else
        echo "[ERROR] Conversion failed for $mp3"
        rm -f "$wav"
    fi
done

echo "Done."