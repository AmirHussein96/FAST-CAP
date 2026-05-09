import os
import torch
import torchaudio
import random
from tqdm import tqdm
from speechbrain.pretrained import SpeakerRecognition
import argparse
import random
import logging
import numpy as np 
from glob import glob

SUPPORTED_EXT = (".wav", ".flac", ".mp3", ".ogg")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load paired source/target audio and compute ViSQOL."
    )
    parser.add_argument("--source-audio-dir", type=str, required=True)
    parser.add_argument("--target-audio-dir", type=str, required=True)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--hf-home", type=str, default="/lustre/fsw/portfolios/edgeai/users/amhussein/cache/HFCACHE")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_audio_dict(root_dir: str):
    """
    Returns:
        dict[id -> AudioSignal]
    """
    audio_dict = {}
    pattern = os.path.join(root_dir, "**", "*")
    # pattern = os.path.join(args.audio_dir, "*")


    audio_files = [
        f for f in glob(pattern, recursive=True)
        if os.path.isfile(f) and f.lower().endswith(SUPPORTED_EXT)
    ]
    for file in sorted(audio_files):
        utt_id = os.path.splitext(os.path.basename(file))[0]
        if "clips_" in utt_id:
            utt_id = utt_id.split("clips_")[-1]
        audio_dict[utt_id] = file

    return audio_dict


def shuffle_and_sample(paired: dict, max_samples: int, seed: int):
    """
    Shuffle a dictionary of paired utterances and return a subsampled version.

    Args:
        paired (dict): Dictionary mapping utterance IDs to paired audio.
        max_samples (int): Maximum number of utterances to keep.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Shuffled and subsampled dictionary.
    """
    items = list(paired.items())
    random.Random(seed).shuffle(items)
    items = items[:max_samples]
    return dict(items)


def build_paired_dict(source_dir: str, target_dir: str):
    """
    Build a paired dictionary of source and target audio.

    Utterances are matched by filename (without extension). Utterances
    missing a corresponding source or target file are skipped.

    Args:
        source_dir (str): Root directory of source audio.
        target_dir (str): Root directory of target audio.
        target_sr (int): Sampling rate for resampling audio.

    Returns:
        dict[str, dict]: Mapping:
            utt_id -> {
                "source": AudioSignal,
                "target": AudioSignal
            }
    """
    logging.info("Loading source audio")
    
    source_audio = load_audio_dict(source_dir)
    logging.info("Loading target audio")
    target_audio = load_audio_dict(target_dir)

    paired = {}
    missing = 0
    for utt_id, tgt_audio_path in target_audio.items():
        if utt_id not in source_audio:
            missing += 1
            continue

        src_audio_path = source_audio[utt_id]

        paired[utt_id] = [src_audio_path, tgt_audio_path]

    logging.info(
        "Paired %d utterances (%d missing targets)",
        len(paired), missing
    )
    return paired


# Function to load and process audio on GPU
def load_and_encode(path, model, device, target_sr):
    waveform, sr = torchaudio.load(path)  # Load audio
    # Select channel
    if waveform.shape[0] > 1:
        waveform = waveform[1:2, :]  # take 2nd channel, keep shape [1, T]
    else:
        waveform = waveform  # already mono
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.to(device)       # Move to GPU
    return model.encode_batch(waveform)  # Extract embedding


def compute_spk_sim(paired_audio, target_sr=16000):
    # Move computations to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Initialize the speaker verification model and move it to GPU
    model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                            savedir="/lustre/fsw/portfolios/edgeai/users/amhussein/pretrained_models", run_opts={"device": device})

    # List to store similarity scores
    similarities = []
    # Compute similarity for each sampled ID
    for file_id in tqdm(paired_audio.keys(), desc="Processing audio pairs"):
        ref_path = paired_audio[file_id][0]
        hyp_path = paired_audio[file_id][1]

        # Load, encode and move to GPU
        ref_emb = load_and_encode(ref_path, model, device, target_sr)
        hyp_emb = load_and_encode(hyp_path, model, device, target_sr)
        ref_emb = torch.nn.functional.normalize(ref_emb, dim=-1)
        hyp_emb = torch.nn.functional.normalize(hyp_emb, dim=-1)

        # Compute cosine similarity (on GPU)
        similarity = torch.nn.functional.cosine_similarity(ref_emb.squeeze(0), hyp_emb.squeeze(0)).item()
        similarities.append(similarity)

    # Compute mean similarity score
    sims = np.asarray(similarities)
    stats = {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
        "num_samples": int(len(sims)),
    }
    print("\n📊 Speaker Similarity Statistics")
    for k, v in stats.items():
        if k != "num_samples":
            print(f"{k:>8}: {v:.4f}")
    print(f"{'samples':>8}: {stats['num_samples']}")

    return stats
  

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Starting Speaker Similarity evaluation")
    logging.info("Source dir: %s", args.source_audio_dir)
    logging.info("Target dir: %s", args.target_audio_dir)

    HF_HOME = args.hf_home
    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_CACHE"] = HF_HOME
    os.environ["HF_DATASETS_CACHE"] = HF_HOME

    paired_audio = build_paired_dict(
        args.source_audio_dir,
        args.target_audio_dir,
    )
    paired_audio = shuffle_and_sample(
        paired_audio,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    logging.info(
        "Evaluating %d utterances (seed=%d)",
        len(paired_audio), args.seed
    )

    stats = compute_spk_sim(paired_audio, args.sampling_rate)


if __name__ == "__main__":
    main()




