from glob import glob
import os
import logging
import argparse
from tqdm import tqdm
import random
import librosa
from speechmos import dnsmos
import numpy as np

SUPPORTED_EXT = (".wav", ".mp3")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load paired source/target audio and compute ViSQOL."
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

# def compute_dnsmos(audio_dir: str, target_sr: int, max_samples: int, seed: int) -> float:
#     """
#     Compute DNSMOS scores for all audio files in a directory.

#     Args:
#         audio_dir (str): Directory containing audio files.
#         target_sr (int): Target sampling rate.
#         max_samples (int): Maximum number of samples to process.
#         seed (int): Random seed for shuffling.

#     Returns:
#         float: Average DNSMOS score across all processed files.
#     """
    
#     # Load audio files
#     pattern = os.path.join(audio_dir, "**", "*")
#     audio_files = [
#         f for f in glob(pattern, recursive=True)
#         if os.path.isfile(f) and f.lower().endswith(SUPPORTED_EXT)
#     ]
#     logging.info("Found %d audio files under %s", len(audio_files), audio_dir)

#     # Shuffle and sample
#     random.Random(seed).shuffle(audio_files)
#     sample_audio_files = audio_files[:max_samples]

#     total_score = []
#     for audio_path in tqdm(sample_audio_files, desc="Computing DNSMOS"):
#         audio, _ = librosa.load(audio_path, sr=target_sr)
#         score = dnsmos.run(audio, sr=target_sr)["ovrl_mos"]
#         total_score.append(score)

#     avg_score = sum(total_score) / len(total_score) if total_score else 0.0
#     return avg_score

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Starting DNSMOS evaluation")
    logging.info("Audio dir: %s", args.audio_dir)
    pattern = os.path.join(args.audio_dir, "**", "*")
    # pattern = os.path.join(args.audio_dir, "*")


    audio_files = [
        f for f in glob(pattern, recursive=True)
        if os.path.isfile(f) and f.lower().endswith(SUPPORTED_EXT)
    ]
    random.Random(args.seed).shuffle(audio_files)
    sample_audio_files = audio_files[:args.max_samples]
    scores = []
    # scores = dnsmos.run(sample_audio_files, sr=16000, verbose= True)
    for path in tqdm(sample_audio_files, desc="Computing DNSMOS"):
        audio, sr = librosa.load(path, sr=None)

        if audio.ndim == 2:
            audio = audio[1]  # or audio[1]
        if sr != args.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sampling_rate, res_type="soxr_vhq")   

        out = dnsmos.run(audio, sr=args.sampling_rate, verbose=False)
        scores.append(out["p808_mos"])
    scores = np.asarray(scores)
    stats = {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "num_samples": int(len(scores)),
    }
    print("\n Audio Quality Statistics")
    for k, v in stats.items():
        if k != "num_samples":
            print(f"{k:>8}: {v:.4f}")
    print(f"{'samples':>8}: {stats['num_samples']}")

if __name__ == "__main__":
    main()