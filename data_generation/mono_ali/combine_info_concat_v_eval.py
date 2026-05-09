import argparse
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator
import os
from lhotse.utils import fastcopy
import logging
import io
import soundfile as sf
from lhotse import Recording, SupervisionSegment, CutSet, MonoCut
import re
import numpy as np



QUOTE_CHARS = '"“”'

def clean_text(text):
    text = re.sub(r'(?:\\"|")(.*?)(?:\\"|")', r'\1', text)
    text = re.sub(r'["“”](.*?)["“”]', r'\1', text)
    text = text.replace('"', '')
    text = text.translate({ord(c): None for c in QUOTE_CHARS})
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# num=2288
# src_manifest=/lustre/fsw/portfolios/edgeai/users/amhussein/data/train/v2/riva_asr_es-US_v1.0/bucket1/sharded_manifests_es-US_en-US/manifest_${num}.jsonl
# src_tar=/lustre/fsw/portfolios/edgeai/users/amhussein/data/train/audio/riva_asr_es-US_v1.0/bucket1/audios_${num}.tar
# src_ctm=/lustre/fsw/portfolios/edgeai/users/amhussein/results/mfa/riva_asr_es-US_v1.0/bucket1/${num}/ctm/words.combined.ctm
# txt_alignment_path=/lustre/fsw/portfolios/edgeai/users/amhussein/users/amhussein/results/awesome_align_clean/riva_asr_es-US_v1.0/bucket1
# tgt_manifest=/lustre/lustre/fsw/portfolios/edgeai/users/amhussein/results/a2flow_tts/riva_asr_es-US_v1.0/bucket1/sharded_manifests_es-US_en-US/manifest_${num}.jsonl
# tgt_tar=/lustre/fsw/portfolios/edgeai/users/amhussein/results/a2flow_tts/riva_asr_es-US_v1.0/bucket1/audios_${num}.tar
# tgt_ctm=/lustre/fsw/portfolios/edgeai/users/amhussein/results/mfa_a2flow/riva_asr_es-US_v1.0/bucket1/${num}/ctm/words.combined.ctm

# num=0
# src_manifest=/lustre/fsw/portfolios/edgeai/users/amhussein/data/cvss/de_en/nemo_prepared_dev/sharded_manifests/manifest_${num}.jsonl
# src_tar=/lustre/fsw/portfolios/edgeai/users/amhussein/data/cvss/de_en/nemo_prepared_dev/audio_${num}.tar
# tgt_manifest=/lustre/fsw/portfolios/edgeai/users/amhussein/results/a2flow_tts/cvss/de_en/dev/sharded_manifests/manifest_${num}.jsonl
# tgt_tar=/lustre/fsw/portfolios/edgeai/users/amhussein/results/a2flow_tts/cvss/de_en/dev/audio_${num}.tar

# python combine_info_concat_v_eval.py --src_tar $src_tar --src_manifest $src_manifest --output_dir debug

def parse_args():
    parser = argparse.ArgumentParser(description="Load data from tar and manifest using lhotse.")
    parser.add_argument('--src_tar', type=str, required=True, help='Path to the tar file containing the data.')
    parser.add_argument('--src_manifest', type=str, required=True, help='Path to the manifest file.')
    parser.add_argument('--min_duration', type=float, default=1.0, help='Minimum duration in seconds.')
    parser.add_argument('--max_duration', type=float, default=30.0, help='Maximum duration in seconds.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--method', type=str, default='alignments', help='Method to combine the cuts.')
    return parser.parse_args()

def remove_extension_from_segment_id(segment):
    return fastcopy(segment, id=os.path.splitext(segment.id)[0])

def load_nemo_tarred_from_dir(manifest_path: str, tar_paths: str) -> CutSet:

    # Initialize iterator
    iterator = LazyNeMoTarredIterator(
                        manifest_path=manifest_path,
                        tar_paths=tar_paths,
                        allow_skipme=False,
                        shuffle_shards=False,
                    )
    return CutSet.from_cuts(iterator)

def recording_from_numpy(waveform: np.ndarray, sr: int, rec_id: str = "rec_in_memory"):
    """
    Create a Lhotse Recording from a NumPy waveform entirely in memory.
    waveform shape: (1, num_samples) or (num_samples,) → mono.
    """
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # (1, num_samples)

    # Encode the waveform into WAV bytes in memory
    buffer = io.BytesIO()
    sf.write(buffer, waveform.T, sr, format="WAV")
    wav_bytes = buffer.getvalue()

    # Use Lhotse's from_bytes()
    recording = Recording.from_bytes(data=wav_bytes, recording_id=rec_id)
    return recording

def add_spk_to_supervisions(cuts, method):
    factor = 1.8 if method == "concat" else 1.4
    new_cuts = []
    for cut in cuts:
        supervisions = []
        total_duration = cut.duration * factor
        id_ = cut.id
        translated_text = clean_text(cut.custom['text_translated_clean'])
        text = cut.supervisions[0].text
        start, src_duration = cut.start, cut.duration
        cut = cut.pad(duration=total_duration, direction="right")
        src_wav = cut.load_audio()
        src_recording = recording_from_numpy(src_wav, cut.sampling_rate, rec_id=f"{id_}")
        supervisions.append(SupervisionSegment(
                    id=f"{id_}_src",
                    recording_id=f"{id_}",
                    start=start,
                    duration=src_duration,
                    channel=0,
                    text=text,
                    speaker="user",
                ))
        supervisions.append(SupervisionSegment(
                    id=f"{id_}_tgt",
                    recording_id=f"{id_}",
                    start=src_duration,
                    duration=total_duration-src_duration,
                    channel=0,
                    text=translated_text,
                    speaker="agent",
                ))
        new_cut = MonoCut(id=id_, start=start, duration=total_duration, channel=0, recording=src_recording, supervisions=supervisions, custom = {'src_duration': src_duration})
        new_cuts.append(new_cut)
    return CutSet.from_cuts(new_cuts)



if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    # Placeholder for loading data using lhotse
    logging.info(f"Tar file: {args.src_tar}")
    logging.info(f"Manifest file: {args.src_manifest}")
    
    src_cuts = load_nemo_tarred_from_dir(args.src_manifest, args.src_tar)
    shard_id = src_cuts[0].custom['shard_id']
    
    # remove the extension from the cut ids
    src_cuts = src_cuts.modify_ids(lambda id: os.path.splitext(id)[0])


    # check before filtering that the number of alignments and cuts are the same
    # assert len(alignment_text_dict) == len(src_cuts), "Mismatch in number of alignments and cuts"

    # filter the cuts to remove the length ratio filter
    # src_cuts = src_cuts.filter(lambda x: x.custom.get('reason') != 'LengthRatioFilter')
    src_cuts = src_cuts.filter(lambda x: x.custom.get('_skipme') != 1)
    # filter out the short and long cuts
    src_cuts = src_cuts.filter(lambda x: args.min_duration <= x.duration <= args.max_duration)
    
   
    # remove the extension from the supervision ids
    src_cuts = src_cuts.map_supervisions(remove_extension_from_segment_id)

    # trim the cuts to the alignments
    src_cuts = src_cuts.trim_to_supervisions(keep_overlapping=False)
 
    new_cuts = add_spk_to_supervisions(src_cuts, args.method)
    # new_cuts = new_cuts.to_eager()
    os.makedirs(args.output_dir, exist_ok=True)
    new_cuts.to_shar(
            args.output_dir,
            shard_size=len(new_cuts),
            shard_offset=shard_id,
            fields={
                'recording': 'wav',
            }
    )
    # loaded_cut = CutSet.from_shar(fields={ "cuts": ["debug/cuts.000000.jsonl.gz"],"recording": ["debug/recording.000000.tar"]})