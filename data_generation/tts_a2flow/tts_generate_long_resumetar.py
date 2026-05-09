"""
TTS Generation Script with Support for Long Audio Segments

This script performs Text-to-Speech (TTS) generation for one utterance at a time, 
with support for chunking long audio based on punctuation. It supports:

- Loading data from Lhotse manifests and tarred audio using `CutSet`.
- Sentence-level segmentation and crossfade-based mel-spectrogram stitching.
- Prompt-based multilingual synthesis using a generator and vocoder.
- Output of generated audio in tar format along with a JSONL manifest.

Typical usage:
    python tts_generate_long.py \
        --tar <tar_path> \
        --manifest <manifest_path> \
        --ctm <ctm_path> \
        --generator_path /pretrained_models/a2f-multilingual-v2 \
        --dp_path /pretrained_models/a2flow-dp-en \
        --vocoder_path /pretrained_models/bigvgan_v2.pt \
        --vocoder_config_path /pretrained_models/bigvgan_v2-config.json \
        --output_dir <results_path> \
        --batch_size 1 \
        --num_workers 1 \
        --max_segment_duration 5.0 \
        --max_pause 1.0 \
        --text_processors <dictionary_path> \
        --ckpt_num 275000 
"""

from eval_utils_multilingual import get_cuts, batch_sample, get_text_processors, get_text, denormalize_mel, split_into_sentences
from utils import set_hparams, HParams, load_checkpoint, load_a2flow, load_dp, load_vocoder
from data_utils import get_lhotse_dataloader, clean_func
from torch.nn.utils.rnn import pad_sequence
import tarfile
import torch
import argparse
import os
from scipy.io.wavfile import write as write_wav
from torch.cuda import amp
import numpy as np
from tqdm import tqdm
import torchaudio
import ndjson
import re
from lhotse import Recording
from lhotse.shar.writers import TarWriter
from lhotse.shar import AudioTarWriter
import io
import librosa

def parse_args():
    parser = argparse.ArgumentParser(description="Load data from tar and manifest using lhotse.")
    parser.add_argument('--tar', type=str, required=True, help='Path to the tar file containing the data.')
    parser.add_argument('--manifest', type=str, required=True, help='Path to the manifest file.')
    parser.add_argument('--ctm', type=str, required=True, help='Path to the ctm file.')
    parser.add_argument('--max_segment_duration', type=float, default=5.0, help='Maximum segment duration.')
    parser.add_argument('--max_pause', type=float, default=1, help='Maximum pause.')
    parser.add_argument('--text_processors', type=str, default='configs/text_processors_config_prondict.json', help='Path to the text processors file.')
    parser.add_argument('--generator_path', type=str, required=True, help='Path to the generator checkpoint.')
    parser.add_argument('--dp_path', type=str, required=True, help='Path to the dp checkpoint.')
    parser.add_argument('--ckpt_num', type=int, default=275000, help='Checkpoint number.')
    parser.add_argument('--vocoder_path', type=str, required=True, help='Path to the vocoder checkpoint.')
    parser.add_argument('--vocoder_config_path', type=str, required=True, help='Path to the vocoder config file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--fp16', type=bool, default=False, help='Use fp16.')   
    parser.add_argument('--save_prompts', type=bool, default=False, help='Save prompts.')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate.')
    parser.add_argument('--sharded_dir', type=str, default=None, help='Sharded manifests directory.')
    parser.add_argument('--clean_tgt_text', type=bool, default=True, help='Clean text.')
    parser.add_argument('--dur_scale', type=float, default=1.0, help='Duration scaling factor.')
    parser.add_argument('--cleanup-wav', action='store_true',help='Remove wav files after creating tar')
    parser.add_argument('--adaptive_dur_scale', action='store_true',help='Dynamic control over the generation speed based on duration ratio')
    return parser.parse_args()


def slow_audio(audio_np, sr, factor):
    """
    Slows audio using librosa phase vocoder (pitch-preserving).
    audio_np: int16 ndarray
    factor: <1 slower, >1 faster
    """
    if factor == 1.0:
        return audio_np
    y = np.ascontiguousarray(audio_np, dtype=np.float32)
    # Librosa time-stretch
    y_stretch = librosa.effects.time_stretch(y, rate=factor)
    return y_stretch.astype(np.float32)

def adaptive_speed(text, min_speed=0.90, max_speed=1.0):
    """
    Compute an adaptive slow-down factor based on sentence length.
    text (str): target text being synthesized
    Returns: speed factor (float)
    """

    words = text.strip().split()
    num_words = len(words)

    # Define thresholds
    if num_words <= 35:
        return min_speed                  # 20% slower
    elif num_words <= 50:
        return 0.95
    else:
        return max_speed                  # no slowdown

def write_txt(text, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# def write_ndjson(data, path):
#     """
#     NDJSON writer that supports:
#     Args:
#         path (str): output .jsonl file path
#         data (list or dict): entries to write/append/update
#     """

#     if not os.path.exists(os.path.dirname(path)):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
    
#     if os.path.exists(path):
#          with open(path, "a", encoding="utf-8") as f:
#             ndjson.dump(data, f, ensure_ascii=True)
#     else:
#          with open(path, "w", encoding="utf-8") as f:
#             ndjson.dump(data, f, ensure_ascii=True)
       
#     return

def write_ndjson(entry, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(ndjson.dumps(entry).strip() + "\n")
    
    

def get_manifest_id(manifest_path):
    match = re.search(r'manifest_(\d+)', manifest_path)
    manifest_num = int(match.group(1)) if match else None
    return manifest_num

def crossfade_mels(mel1, mel2, silence, overlap=4):
    """
    Crossfade two mel spectrograms of shape [1, 80, T] using linear fade over `overlap` frames.
    Silence: [1, 80, silence_frames]
    """
    device = mel1.device
    fade_out = torch.linspace(1.0, 0.0, overlap).view(1, -1).to(device)
    
    fade_in = 1.0 - fade_out
    fade_in = fade_in.to(device)
    # window = torch.hamming_window(overlap * 2, periodic=False, device=device)
    # fade_out = window[:overlap].view(1, -1)  # First half (1 → 0)
    # fade_in = window[overlap:].view(1, -1)   # Second half (0 → 1)
    if silence is not None:
        mel1 = torch.cat([mel1, silence], dim=1)
        mel2 = torch.cat([silence, mel2], dim=1)
    mel1_end = mel1[:, -overlap:] * fade_out 
    mel2_start = mel2[:, :overlap] * fade_in 
    blended = mel1_end + mel2_start

    return torch.cat([mel1[:, :-overlap], blended, mel2[:, overlap:]], dim=1)

def is_valid(cut, max_expected_duration=60.0, min_duration=1.0, max_duration=60.0):
    """
    Determine whether a speech segment (cut) is valid based on its actual duration
    and an estimated target speech duration after translation.

    Parameters
    ----------
    cut : lhotse.Cut
    max_expected_duration : float, optional (default: 60.0)
        Maximum estimated duration of target audio (in seconds) 
    min_duration : float, optional (default: 1.0)
        Minimum duration of the source audio (in seconds).
    max_duration : float, optional (default: 60.0)
        Maximum  duration of the source audio (in seconds).

    Returns
    -------
    bool
        True if the cut passes both duration checks, False otherwise.
    """
    # Remove spaces from both texts
    txt = cut.custom.get('text_translated_clean')
    if args.clean_tgt_text and txt:
        txt = clean_func(cut.custom['target_lang'])(txt)
    if not txt:
        return False
    source_chars = len(cut.supervisions[0].text.replace(" ", ""))
    translated_chars = len(txt.replace(" ", ""))
    
    # Compute source characters per second
    cps_source = source_chars / cut.duration
    
    # Compute expected duration to speak translated text at source CPS
    expected_duration = translated_chars / cps_source
    
    # Filter out if expected duration exceeds 20s
    return (
        min_duration <= cut.duration <= max_duration
        and expected_duration <= max_expected_duration
    )

def finished_files(expected_ids, wav_dir):
    expected_ids = set(expected_ids)
    generated_wav_ids = {os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith(".wav")}
    return expected_ids - generated_wav_ids


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_prompts:
        os.makedirs(f"{args.output_dir}/prompts", exist_ok=True)
    manifest_id = get_manifest_id(args.manifest)
    # Load text processors
    text_processors = get_text_processors(args.text_processors)
    # Load generator, dp and vocoder
    hps_generator = set_hparams(args.generator_path)
    hps_dp = set_hparams(args.dp_path)
    generator = load_a2flow(args.generator_path, hps_generator, n_language=len(text_processors), num=args.ckpt_num).cuda().eval()        
    dp = load_dp(args.dp_path, hps_dp, n_language=len(text_processors)).cuda().eval()
    vocoder = load_vocoder(args.vocoder_config_path, args.vocoder_path).cuda()
    device = next(generator.parameters()).device

    # Get cuts
    cuts = get_cuts(args.manifest, args.tar, args.ctm, args.max_segment_duration, args.max_pause, allow_skipme=False)
    # filter out cuts that does not have alignments
    cuts = cuts.filter(lambda c: all(getattr(s, "alignment", None) for s in c.supervisions))
    cuts = cuts.filter(lambda x: x.id != "_es_voxpopuli_train_20191218-0900-PLENARY-es_20191218-19:54:47_0")
    # cuts = cuts.filter(lambda x: x.custom.get('reason') != 'LengthRatioFilter')
    # cuts = cuts.filter(lambda x: x.duration <= 18 and x.duration >= 1)
    
    cuts_valid = cuts.filter(is_valid)    # filter long source and long expected target audios
    
    cuts_valid = cuts_valid.resample(args.sample_rate)
    ids = [c.id for c in cuts_valid]
    # create dataloader
    dataloader = get_lhotse_dataloader(cuts_valid, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers, 
                                        max_pause=args.max_pause, 
                                        max_segment_duration=args.max_segment_duration,
                                        text_processors=text_processors)
    # for each batch from dataloader perform batch inference ie "batch sample"
    # tar_writer = TarWriter(f"{args.output_dir}/audio_{manifest_id}.tar")
    # tar_writer = AudioTarWriter(f"{args.output_dir}/audios_{manifest_id}.tar", shard_size=None, format="wav")
    silence = torch.full((80, 3), fill_value=-16.5129, device=device)
    for batch in tqdm(dataloader):
        if batch is None:
            continue
        for idx in range(len(batch['translated_text'])):
            wav_path = f"{args.output_dir}/{manifest_id}/{batch['id'][idx]}.wav"
            if os.path.exists(wav_path):
                print(f"Audio exists: {wav_path}")
                continue
            try:
                text_gens = [batch['translated_text'][idx]]
                tgt_lang = batch['target_lang'][idx]
                src_lang = batch['source_lang'][idx]
                dur_factors = batch['dur_factors'][idx]
                src_dur = batch['src_dur'][idx]
                audio_id = batch['id'][idx]
                prompt_text = batch['text'][idx]
                prompt_lang = batch['source_lang'][idx]
                expected_duration = batch['expected_duration'][idx]
                prompt_mel_lengths = batch['mel_lens'][idx].unsqueeze(0).to(device)
                prompt_mel = batch['mels'].to(device)
                prompt_language_id = torch.tensor([list(text_processors.keys()).index(batch['source_lang'][idx])],dtype=torch.long, device=device)  
                synth_language_id = torch.tensor([list(text_processors.keys()).index(batch['target_lang'][idx])], dtype=torch.long, device=device)
                
                # 
                # get the prompt length
                prompt_norm = [get_text(
                    prompt_text=prompt_text, 
                    prompt_lang=prompt_lang, 
                    text_processors=text_processors
                    )]
                prompt_lengths = torch.tensor([t.size(0) for t in prompt_norm], dtype=torch.long, device=device)

                
                if args.adaptive_dur_scale:
                    if dur_factors >= 2 :
                        dur_scale = dur_factors - 0.5
                    elif dur_factors >= 1.5 and dur_factors < 2:
                        dur_scale = dur_factors - 0.3
                    elif dur_factors >= 1.2 and dur_factors < 1.5:
                        dur_scale = dur_factors - 0.2
                    else:
                        dur_scale = 1
                else:
                    dur_scale = args.dur_scale
                # check if the expected duration of the TTS audio is too long 
                expected_duration *= dur_scale      
                if expected_duration >= 19:
                    # split target text based on punctuations, if no punctuation exist use max_words
                    text_gens = split_into_sentences(text_gens[0], max_words=20)
                mel_segments = []
                y_length_tot = 0
                for text_gen in text_gens:
                    # convert target text added to it the prompt into ids
                    text_norm = [get_text(
                        prompt_text=prompt_text, 
                        prompt_lang=prompt_lang, 
                        synth_text=text_gen, 
                        synth_lang=tgt_lang, 
                        text_processors=text_processors) ]
                    
                    text_norm_lengths = torch.tensor([t.size(0) for t in text_norm], dtype=torch.long, device=device)
                    text_norm = pad_sequence(text_norm, batch_first=True, padding_value=0).to(device)
                    # this is just to get the length in the target language to compute the expected duration later
                    text_dp_norm = [get_text(synth_text=text_gen, 
                            synth_lang=batch['target_lang'][idx], 
                            text_processors=text_processors)]
                    x_dp_lengths = torch.tensor([t.size(0) for t in text_dp_norm], dtype=torch.long, device=device)
                    
                    with amp.autocast(enabled=args.fp16):
                        y_length_dp = (prompt_mel_lengths / prompt_lengths) * x_dp_lengths
                        
                        ratio = prompt_mel_lengths / prompt_lengths
                        # print(f"{audio_id}: duration ration {ratio}")

                        print(f"id: {audio_id}")
                        print(f"dur scale: {dur_scale}")
                        print(f"src_dur: {src_dur}, tgt_expected_duration: {expected_duration}")
                        y_length = (y_length_dp * dur_scale).round().long()
                        # print(f"x_dp_lengths: {x_dp_lengths}")
                        # print(f"y_lengths: {y_lengths}")
                        # the values for n_timesteps and alpha are taken from the paper https://openreview.net/pdf?id=e2p1BWR3vq
                        _, y_dec = generator.infer_batch(
                            text_norm, text_norm_lengths, p=prompt_mel, p_lengths=prompt_mel_lengths, y_lengths=y_length, n_timesteps=32,
                            gradient_scale=2, alpha=3.0,
                            prompt_language_id=prompt_language_id, synth_language_id=synth_language_id
                        )
                        mel_segments += y_dec

                        y_length_tot += y_length

                # mel_cat = torch.cat(mel_segments, dim=1).unsqueeze(0) 

                mel_cat = mel_segments[0]
                for seg in mel_segments[1:]:
                    mel_cat = crossfade_mels(mel_cat, seg, silence=silence, overlap=8) 
        
                mel_cat = denormalize_mel(mel_cat).unsqueeze(0)
                with torch.no_grad():
                    # mel_cat = torch.cat(mel_segments, dim=1).unsqueeze(0)
                    audio = [vocoder.forward(y.unsqueeze(0).detach()) for y in mel_cat]
                    audio_int16 = [(a.detach().cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16) for a in audio]
                # if args.extra_speed_audio_factor:
                #     speed_factor = adaptive_speed(text_gen)
                #     audio_int16 = [ slow_audio(audio, args.sample_rate, speed_factor) for audio in audio_int16]
                # audio_int16 = [(a.detach().cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16) for a in audio_float]

                    # audio = [(a.detach().cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16) for a in audio]
                # for idx, text_gen in enumerate(batch['translated_text']):
                    
                audio_np = audio_int16[idx]
                # buffer = io.BytesIO()
                # write_wav(buffer, rate=args.sample_rate, data=audio_np)
                # buffer.seek(0)  # reset pointer to beginning
                # tar_writer.write(f"{audio_id}.wav", buffer)  # pass buffer, not buffer.getvalue()
                # recording = Recording.from_bytes(
                #     data=buffer.getvalue(),
                #     id=audio_id
                # )
                # tar_writer.write(key=audio_id, value=audio_np, sampling_rate=args.sample_rate, manifest=recording)
                if not os.path.exists(os.path.dirname(wav_path)):
                    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

                # print(f"text: {text_gens}")
                write_wav(wav_path, rate=args.sample_rate, data=audio_np)
                target_manifest_entries = {
                    "audio_filepath": f"{audio_id}.wav",
                    "duration": audio_np.shape[0] / args.sample_rate,
                    "source_lang": 'es-US' if batch['target_lang'][idx] == 'es_MX' else batch['target_lang'][idx].replace('_', '-'),
                    "taskname": "translate",
                    "text": batch['translated_text'][idx],
                    "shard_id": cuts[0].custom.get('shard_id')
                }
                write_ndjson([target_manifest_entries], f"{args.output_dir}/{args.sharded_dir}/manifest_{manifest_id}.jsonl")
            except Exception as e:
                audio_id = batch['id'][idx]
                print(f"❌ Skipped {audio_id} due to error: {e}")
                continue
            
        if args.save_prompts:
            for idx, transcript in enumerate(batch['text']):
                write_txt(transcript, f"{args.output_dir}/prompts/{batch['id'][idx]}.txt")
                torchaudio.save(f"{args.output_dir}/prompts/{batch['id'][idx]}.wav", batch['audio'][idx], args.sample_rate)

    
    wav_dir = f"{args.output_dir}/{manifest_id}"
    missing = finished_files(ids, wav_dir)
    if missing:
        print(f"❌ Not all files written. Missing {len(missing)} files:")
    else:
        print("✔ All files written. Proceeding to tar.")
        tar_path = f"{args.output_dir}/audio_{manifest_id}.tar"
        if not os.path.exists(tar_path):
            print(f"Creating tar archive at: {tar_path}")

            with tarfile.open(tar_path, "w") as tar:
                for fname in os.listdir(wav_dir):
                    if fname.endswith(".wav"):
                        tar.add(os.path.join(wav_dir, fname), arcname=fname)

            print("✔ TAR complete.")
        else:
            print(f"Tar file exists: {tar_path}")

        if args.cleanup_wav:
            print(f"🧹 cleanup=True → removing directory {wav_dir} ...")
            import shutil
            shutil.rmtree(wav_dir, ignore_errors=True)
            print("✔ WAV directory removed.")