import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import io
import soundfile as sf

from lhotse import CutSet, MonoCut, Recording
from lhotse.utils import fastcopy


# ---------------------------
# Utilities
# ---------------------------

def recording_from_numpy(waveform: np.ndarray, sr: int, rec_id: str):
    """
    Create a Lhotse Recording fully in memory from a numpy waveform.
    waveform shape: (1, T) or (T,)
    """
    if waveform.ndim == 1:
        waveform = waveform[None, :]

    buffer = io.BytesIO()
    sf.write(buffer, waveform.T, sr, format="WAV")
    wav_bytes = buffer.getvalue()

    return Recording.from_bytes(
        data=wav_bytes,
        recording_id=rec_id,
    )


def compute_speaker_durations(cuts: CutSet, speaker_key: str):
    speaker_durations = defaultdict(float)
    for cut in cuts:
        spk = cut.custom.get(speaker_key)
        if spk is not None:
            speaker_durations[spk] += cut.duration
    return speaker_durations


def concat_cuts_per_speaker(
    cuts: CutSet,
    max_duration: float,
    speaker_key: str,
):
    """
    Concatenate cuts per speaker until max_duration is reached.
    Returns a CutSet of new MonoCuts.
    """
    spk2cuts = defaultdict(list)
    for cut in cuts:
        spk = cut.custom.get(speaker_key)
        if spk is not None:
            spk2cuts[spk].append(cut)

    new_cuts = []

    for spk, spk_cuts in spk2cuts.items():
        spk_cuts = sorted(spk_cuts, key=lambda c: c.start)

        sr = spk_cuts[0].sampling_rate
        buf_audio = []
        buf_supervisions = []
        buf_dur = 0.0
        offset = 0.0

        def flush():
            nonlocal buf_audio, buf_supervisions, buf_dur, offset
            if not buf_audio:
                return

            wav = np.concatenate(buf_audio, axis=1)
            rec_id = f"{spk}_{len(new_cuts)}"

            recording = recording_from_numpy(wav, sr, rec_id)

            new_cut = MonoCut(
                id=rec_id,
                start=0.0,
                duration=buf_dur,
                channel=0,
                recording=recording,
                supervisions=buf_supervisions,
                custom={"origin_spk": spk},
            )
            new_cuts.append(new_cut)

            buf_audio = []
            buf_supervisions = []
            buf_dur = 0.0
            offset = 0.0

        for cut in spk_cuts:
            if cut.sampling_rate != sr:
                cut = cut.resample(sr)

            if buf_dur + cut.duration > max_duration:
                flush()

            audio = cut.load_audio()  # (1, T)
            buf_audio.append(audio)

            for sup in cut.supervisions:
                buf_supervisions.append(
                    fastcopy(
                        sup,
                        start=sup.start + offset,
                        recording_id=f"{spk}_{len(new_cuts)}",
                    )
                )

            offset += cut.duration
            buf_dur += cut.duration

        flush()

    return CutSet.from_cuts(new_cuts)


# ---------------------------
# Main
# ---------------------------

def main(args):
    data_path = Path(args.data_dir)
    print(f"Loading SHAR data from: {data_path}")

    cuts = CutSet.from_shar(in_dir=data_path)

    # 1) unique speakers
    uniq_speakers = set([cut.custom['origin_spk'] for cut in cuts])

    print(f"Unique speakers: {len(uniq_speakers)}")

    # 2) speaker durations
    speaker_durations = compute_speaker_durations(
        cuts, speaker_key="origin_spk"
    )
    if args.min_speaker_duration > 0:
        speakers_ge_thresh = {
            spk: dur
            for spk, dur in speaker_durations.items()
            if dur >= args.min_speaker_duration
        }
    else:
        speakers_ge_thresh = uniq_speakers
    print(
        f"Speakers with >= {args.min_speaker_duration}s: "
        f"{len(speakers_ge_thresh)}"
    )

    # 3) filter cuts to long speakers only
    cuts_long = cuts.filter(
        lambda c: c.custom.get("origin_spk") in speakers_ge_thresh
    )

    # 4) concatenate
    print("Concatenating cuts per speaker...")
    new_cuts = concat_cuts_per_speaker(
        cuts_long,
        max_duration=args.max_concat_duration,
        speaker_key="origin_spk",
    )

    print(f"Created {len(new_cuts)} concatenated cuts")

    # 5) save to SHAR
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving SHAR to: {output_dir}")
    new_cuts.to_shar(
        output_dir,
        shard_size=len(new_cuts),
        fields={
            "recording": "wav",
        },
    )

    print("Done.")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate CVSS speakers and export to Lhotse SHAR"
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Input SHAR directory (e.g. cvss/fr-FR_en-US/test)",
    )

    parser.add_argument(
        "--min-speaker-duration",
        type=float,
        default=40.0,
        help="Minimum total speaker duration (seconds)",
    )
    parser.add_argument(
        "--max-concat-duration",
        type=float,
        default=100.0,
        help="Maximum duration per concatenated cut (seconds)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output SHAR directory",
    )

    args = parser.parse_args()
    main(args)
