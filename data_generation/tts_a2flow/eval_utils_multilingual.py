import json
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.functional as TAF
from tts_text_processing.text_processing_dict import TextProcessing
import datetime as dt
import IPython.display as ipd
from torch.cuda import amp
from bigvgan.meldataset import mel_spectrogram
# from transformers import Wav2Vec2Processor, HubertForCTC, AutoProcessor, AutoModelForSpeechSeq2Seq
from pathlib import Path
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator
from lhotse import CutSet
import os
from lhotse.supervision import AlignmentItem
from collections import defaultdict
from lhotse.utils import fastcopy
import codecs
import re


def split_into_sentences(text, max_words=30, min_words=10):
    # Step 1: Split by punctuation
    raw_sentences = re.split(r'(?<=[.!])\s+', text.strip())
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Step 2: Merge short single-word sentences with the previous one
    merged_sentences = []
    for s in raw_sentences:
        word_count = len(s.split())
        if word_count < min_words and merged_sentences:
            # Merge with previous sentence
            merged_sentences[-1] += ' ' + s
        else:
            merged_sentences.append(s)

    # Step 3: Fallback if splitting didn't work or only 1 sentence
    if len(merged_sentences) > 1:
        return merged_sentences

    # Step 4: Fallback to chunking by max_words
    words = text.strip().split()
    if len(words) <= max_words:
        return [text.strip()]

    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk.strip())

    return chunks


def get_duration(n_samples, sampling_rate, rounding=3):
    return np.round(n_samples / sampling_rate, 3)


def adjust_minimum_loudness(audio_array, sample_rate, min_loudness):
    """
    Adjusts the loudness of an audio array to meet a minimum loudness threshold while preserving dynamics.

    Args:
        audio_array: numpy array of audio samples in [-1, 1] range
        sample_rate: sampling rate of the audio in Hz
        min_loudness: minimum target loudness in dB FS RMS

    Returns:
        adjusted_audio_array: processed audio with adjusted loudness
    """
    # Calculate RMS value in dB FS (Full Scale)
    # Use small epsilon to avoid log of zero
    eps = 1e-10
    rms = np.sqrt(np.mean(audio_array**2))
    current_loudness = 20 * np.log10(max(rms, eps))

    # Calculate required gain while accounting for headroom
    if current_loudness < min_loudness:
        # Add headroom to prevent peaks from exceeding [-1, 1]
        # Find the peak value to calculate available headroom
        peak_value = np.max(np.abs(audio_array))
        max_gain_before_clipping = -20 * np.log10(peak_value) if peak_value > eps else 0

        # Calculate desired gain
        desired_gain = min_loudness - current_loudness

        # Limit gain to prevent clipping
        applied_gain = min(desired_gain, max_gain_before_clipping)

        # Convert dB gain to linear scale
        gain_factor = 10**(applied_gain / 20)

        # Apply gain to the audio array
        adjusted_audio_array = audio_array * gain_factor
    else:
        # If already loud enough, no adjustment needed
        adjusted_audio_array = audio_array

    return adjusted_audio_array


def adjust_loudness(audio_array, sample_rate, target_loudness):
    # Calculate current loudness in dB (RMS value in dB)
    rms = np.sqrt(np.mean(audio_array**2))
    current_loudness = 20 * np.log10(rms) if rms > 0 else -float("inf")

    # Calculate the required gain to reach the target loudness
    gain = target_loudness - current_loudness
    gain_factor = 10**(gain / 20)  # Convert dB gain to linear scale

    # Apply gain to the audio array
    adjusted_audio_array = audio_array * gain_factor

    # Ensure the adjusted audio does not exceed [-1, 1] range
    adjusted_audio_array = np.clip(adjusted_audio_array, -1.0, 1.0)

    return adjusted_audio_array


def get_item(text_list, wav, mel, transcript=''):
    return [{'text': text, 'wav': wav, 'p': mel, 'transcript': transcript} for text in text_list]


# def get_s2t(use_whisper):
#     if use_whisper:
#         s2t_processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
#         s2t = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
#         print("Load Whisper to transcribe the prompt")
#     else:
#         s2t_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
#         s2t = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
#         print("Load HuBERT-L to transcribe the prompt")
#     return s2t, s2t_processor


# def get_prompt(sample_path, num_second, transcript=None, use_whisper=True):
#     # downsample_16k = torchaudio.transforms.Resample(22050, 16000)
#     s2t, s2t_processor = get_s2t(use_whisper)
#     s2t = s2t.eval().cuda()

#     wav, _ = librosa.load(sample_path, sr=22050)
#     wav = torch.FloatTensor(wav)[:int(num_second * 22050)]
#     #print("Prompt Audio")
#     #ipd.display(ipd.Audio(wav, rate=22050))
#     if wav.max() <= 1.5 and wav.min() >= -1.5:
#         wav = wav
#     else:
#         wav = wav + torch.rand_like(wav)
#         wav = wav / 32768.0
#     wav = wav.unsqueeze(0)
#     mel = mel_spectrogram(wav, 1024, 80, 22050, 256, 1024, 0., 11025, center=False).squeeze()
#     # if transcript is None:
#     #     # if transcript is None, use ASR model to extract transcript from the prompt
#     #     wav_16k = downsample_16k(wav)
#     #     wav_16k_transcript = torch.cat((torch.zeros((1, 400)), wav_16k, torch.zeros((1, 400))), dim=1)
#     #     if use_whisper:
#     #         transcript = get_transcript(wav_16k_transcript.squeeze(0).cuda(), 'english', s2t, s2t_processor, sr=16000, model_name='whisper')
#     #     else:
#     #         transcript = get_transcript(wav_16k_transcript.squeeze(0).cuda(), 'english', s2t, s2t_processor, sr=16000, model_name='hubert-large')
#     #     print(f"Transcript of the prompt: {transcript}")
#     return wav, mel, transcript


@torch.no_grad()
def sample(args, item_list, text_processors, generator, dp, vocoder):
    all_audio = []
    for i, item in enumerate(item_list):
        transcript = item['transcript']
        text_gen = item['text']

        wav_gt = item['wav'].unsqueeze(0).cuda()

        text_dp_norm = get_text(
            synth_text=text_gen, 
            synth_lang=args.synth_lang, 
            text_processors=text_processors
        )
        text_norm = get_text(
            prompt_text=transcript, 
            prompt_lang=args.prompt_lang, 
            synth_text=text_gen, 
            synth_lang=args.synth_lang, 
            text_processors=text_processors
        )
        x_dp = text_dp_norm.unsqueeze(0).cuda()
        x_dp_lengths = torch.LongTensor([x_dp.shape[-1]]).cuda()
        x = text_norm.unsqueeze(0).cuda()
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
        prompt_language_id = torch.LongTensor([list(text_processors.keys()).index(args.prompt_lang)]).cuda()
        synth_language_id = torch.LongTensor([list(text_processors.keys()).index(args.synth_lang)]).cuda()
        p = item['p'].unsqueeze(0).cuda()
        p_lengths = torch.LongTensor([p.shape[-1]]).cuda()
        p = normalize_mel(p)
 
        transcript_norm = get_text(
            prompt_text=transcript, 
            prompt_lang=args.prompt_lang, 
            text_processors=text_processors
        )
        transcript_cuda = transcript_norm.unsqueeze(0).cuda()
        transcript_lengths = torch.LongTensor([transcript_cuda.shape[-1]]).cuda()

        full_text_norm = get_text(
            prompt_text=transcript,
            synth_text=text_gen, 
            prompt_lang=args.prompt_lang, 
            synth_lang=args.synth_lang,
            text_processors=text_processors
        )
        full_text_cuda = full_text_norm.unsqueeze(0).cuda()
        full_text_lengths = torch.LongTensor([full_text_cuda.shape[-1]]).cuda()

        # # print(x_dp)
        # # print(transcript_cuda)
        # print(x)
        # # print(full_text_cuda)

        # ending = [0, 0]
        # if not torch.eq(transcript_cuda[:, -2], 8):
        #     ending = [8] + ending
        # tensor_size = transcript_cuda.size(0)
        # ending = torch.tensor(ending, device=transcript_cuda.device)
        # ending = ending.unsqueeze(0).expand(tensor_size, -1).cuda()

        # composite_text_cuda = transcript_cuda + ending + x_dp
        # composite_text_cuda = torch.cat([transcript_cuda, ending, x_dp], dim=1)
        # print(composite_text_cuda)
        # print(x == composite_text_cuda)


        if args.display_audio:
            print(f"Transcript of Prompt: {transcript}")
            print("Prompt Audio")
            ipd.display(ipd.Audio(item['wav'], rate=22050))

        torch.cuda.synchronize()
        t = dt.datetime.now()
        with amp.autocast(enabled=args.fp16):
            # print("Using regression len formula. Comparison:")
            # y_lengths_dp = dp(x_dp, x_dp_lengths, p, p_lengths, language_id=synth_language_id)
            # print(y_lengths_dp, type(y_lengths_dp))

            y_lengths_dp = (p_lengths / transcript_lengths) * x_dp_lengths
            y_lengths_dp = torch.tensor([y_lengths_dp]).cuda(0)
            # print(y_lengths_dp, type(y_lengths_dp))

            # #  -0.1623 * prompt_mel_siwe + 3.0583 * textToGen_phonemes_size + -0.6582 * phonemes_transcript_size(-1) + 377.5714
            # y_lengths_dp = -0.1623 * x_dp_lengths + 3.0583 * transcript_lengths + -0.6582 * p_lengths[-1] + 200.5714
            # y_lengths_dp = torch.tensor([y_lengths_dp]).cuda(0)
            # print(y_lengths_dp, type(y_lengths_dp))
            
            dur_scale = getattr(args, "dur_scale", 1)
            print(f"duration_scale: {dur_scale}")
            y_lengths = y_lengths_dp * dur_scale
            print("gen_len",y_lengths)
            _, y_dec = generator.infer(
                x, x_lengths, p=p, p_lengths=p_lengths, y_lengths=y_lengths, n_timesteps=args.timesteps,
                gradient_scale=args.gradient_scale, alpha=args.alpha,
                prompt_language_id=prompt_language_id, synth_language_id=synth_language_id
            )
        y_dec = denormalize_mel(y_dec)
        torch.cuda.synchronize()
        t1 = (dt.datetime.now() - t).total_seconds()

        print(f"Text input: {text_gen}")
        print("Generated Audio")
        with torch.no_grad():
            audio = vocoder.forward(y_dec.detach())
            audio = (audio.cpu().squeeze(1).clamp(-1, 1).numpy() * 32768).astype(np.int16)
            audio_obj = ipd.Audio(audio, rate=22050)
            if args.display_audio:
                ipd.display(audio_obj)
        torch.cuda.synchronize()
        t2 = (dt.datetime.now() - t).total_seconds()
        #print(f'A2Flow RTF: {t1 * 22050 / audio.shape[-1]}')
        #print(f'BigVGAN RTF: {(t2-t1) * 22050 / audio.shape[-1]}')
        #print(f'Total RTF: {t2 * 22050 / audio.shape[-1]}')
        #print(f'Total Time: {t1}sec')
        print('===============================================================')
        all_audio.append(audio_obj)
        return audio_obj
    
    # return all_audio

def load_ctm_to_dict(ctm_path: str, id2dur: dict) -> dict:
    """
    Load alignment dict from ctm file and return a dict with utterance id as key and list of AlignmentItem as value
    """
    alignment_dict = defaultdict(list)
    skipped = []
    with open(ctm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id, _, start, duration, symbol = parts[:5]
            if utt_id not in id2dur:
                print(f"Utterance {utt_id} not found in cuts")
                if utt_id not in skipped:
                    skipped.append(utt_id)
                continue
            
            alignment_dict[utt_id].append(
                AlignmentItem(symbol=symbol, start=float(start), duration=float(duration))
            )
            if alignment_dict[utt_id][-1].end > id2dur[utt_id]:
                # fix the last alignment item to be the duration of the cut
                alignment_dict[utt_id][-1] = AlignmentItem(symbol=alignment_dict[utt_id][-1].symbol, start=alignment_dict[utt_id][-1].start, duration=id2dur[utt_id] - alignment_dict[utt_id][-1].start)
    print(f"Skipped {len(skipped)} utterances")
    return alignment_dict

def get_add_alignment_fn(alignment_dict):
    def add_alignment(sup):
        if sup.id in alignment_dict:
            return sup.with_alignment("word", alignment_dict[sup.id])
        return sup
    return add_alignment

def remove_extension_from_segment_id(segment):
    """
    Remove extension from segment id
    """
    return fastcopy(segment, id=os.path.splitext(segment.id)[0])

def load_nemo_tarred_from_dir(manifest_path: str, tar_paths: str, allow_skipme: bool) -> CutSet:
    """
    Load cuts from tarred files
    """
    # Initialize iterator
    iterator = LazyNeMoTarredIterator(
                        manifest_path=manifest_path,
                        tar_paths=tar_paths,
                        allow_skipme=allow_skipme,
                        skip_missing_manifest_entries=True,
                    )
    return CutSet(iterator)

def get_cuts(manifest_path: str, tar_paths: str, ctm_path: str, max_segment_duration: float = 5.0, max_pause: float = 0.5, allow_skipme=False) -> CutSet:
    # Load cuts from tarred files
    cuts = load_nemo_tarred_from_dir(manifest_path, tar_paths, allow_skipme=allow_skipme)
    # Remove extension from cut ids
    cuts = cuts.modify_ids(lambda id: os.path.splitext(id)[0])
    # Load alignment dict from ctm file
    id2dur = {c.id: c.duration for c in cuts}
    alignment_dict = load_ctm_to_dict(ctm_path, id2dur)
    # Remove extension from segment ids
    cuts = cuts.map_supervisions(remove_extension_from_segment_id)
    # Add alignment to supervisions
    cuts = cuts.map_supervisions(get_add_alignment_fn(alignment_dict))
    cuts = cuts.trim_to_supervisions(keep_overlapping=False)
    return cuts

@torch.no_grad()
def batch_sample(args, item_list, text_processors, generator, dp, vocoder, output_dir):
    all_audio = []
    for i, item in enumerate(item_list):
        transcript = item['transcript']
        text_gen = item['text']

        text_dp_norm = get_text(
            synth_text=text_gen, 
            synth_lang=args.synth_lang, 
            text_processors=text_processors
        )
        text_norm = get_text(
            prompt_text=transcript, 
            prompt_lang=args.prompt_lang, 
            synth_text=text_gen, 
            synth_lang=args.synth_lang, 
            text_processors=text_processors
        )
        x_dp = text_dp_norm.unsqueeze(0).cuda()
        x_dp_lengths = torch.LongTensor([x_dp.shape[-1]]).cuda()
        x = text_norm.unsqueeze(0).cuda()
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
        prompt_language_id = torch.LongTensor([list(text_processors.keys()).index(args.prompt_lang)]).cuda()
        synth_language_id = torch.LongTensor([list(text_processors.keys()).index(args.synth_lang)]).cuda()
        p = item['p'].unsqueeze(0).cuda()
        p_lengths = torch.LongTensor([p.shape[-1]]).cuda()
        p = normalize_mel(p)
 
        transcript_norm = get_text(
            prompt_text=transcript, 
            prompt_lang=args.prompt_lang, 
            text_processors=text_processors
        )
        transcript_cuda = transcript_norm.unsqueeze(0).cuda()
        transcript_lengths = torch.LongTensor([transcript_cuda.shape[-1]]).cuda()

        full_text_norm = get_text(
            prompt_text=transcript,
            synth_text=text_gen, 
            prompt_lang=args.prompt_lang, 
            synth_lang=args.synth_lang,
            text_processors=text_processors
        )

        if args.display_audio:
            print(f"Transcript of Prompt: {transcript}")
            print("Prompt Audio")
            ipd.display(ipd.Audio(item['wav'], rate=22050))

        torch.cuda.synchronize()
        t = dt.datetime.now()
        with amp.autocast(enabled=args.fp16):
            # print("Using regression len formula. Comparison:")
            # y_lengths_dp = dp(x_dp, x_dp_lengths, p, p_lengths, language_id=synth_language_id)
            # print(y_lengths_dp, type(y_lengths_dp))

            y_lengths_dp = (p_lengths / transcript_lengths) * x_dp_lengths
            y_lengths_dp = torch.tensor([y_lengths_dp]).cuda(0)
            # print(y_lengths_dp, type(y_lengths_dp))

            # #  -0.1623 * prompt_mel_siwe + 3.0583 * textToGen_phonemes_size + -0.6582 * phonemes_transcript_size(-1) + 377.5714
            # y_lengths_dp = -0.1623 * x_dp_lengths + 3.0583 * transcript_lengths + -0.6582 * p_lengths[-1] + 200.5714
            # y_lengths_dp = torch.tensor([y_lengths_dp]).cuda(0)
            # print(y_lengths_dp, type(y_lengths_dp))
            
            dur_scale = getattr(args, "dur_scale", 1)
            print(f"duration_scale: {dur_scale}")
            y_lengths = y_lengths_dp * dur_scale
            print("gen_len",y_lengths)
            _, y_dec = generator.infer(
                x, x_lengths, p=p, p_lengths=p_lengths, y_lengths=y_lengths, n_timesteps=args.timesteps,
                gradient_scale=args.gradient_scale, alpha=args.alpha,
                prompt_language_id=prompt_language_id, synth_language_id=synth_language_id
            )
        y_dec = denormalize_mel(y_dec)
        torch.cuda.synchronize()
        t1 = (dt.datetime.now() - t).total_seconds()

        print(f"Text input: {text_gen}")
        print("Generated Audio")
        with torch.no_grad():
            audio = vocoder.forward(y_dec.detach())
            audio = (audio.cpu().squeeze(1).clamp(-1, 1).numpy() * 32768).astype(np.int16)
            audio_obj = ipd.Audio(audio, rate=22050)
            if args.display_audio:
                ipd.display(audio_obj)
        torch.cuda.synchronize()
        t2 = (dt.datetime.now() - t).total_seconds()
        #print(f'A2Flow RTF: {t1 * 22050 / audio.shape[-1]}')
        #print(f'BigVGAN RTF: {(t2-t1) * 22050 / audio.shape[-1]}')
        #print(f'Total RTF: {t2 * 22050 / audio.shape[-1]}')
        #print(f'Total Time: {t1}sec')
        print('===============================================================')
        all_audio.append(audio_obj)
        return audio_obj

def normalize_mel(mel):
    mel_offset = -5.884
    mel_scale = 2.261
    mel = (mel - mel_offset) / mel_scale
    return mel

def denormalize_mel(mel):
    mel_offset = -5.884
    mel_scale = 2.261
    mel = mel * mel_scale + mel_offset
    return mel

def clean_text(text):
    text = text.replace('\\"', '"')
    return text

def get_text(prompt_text=None, prompt_lang=None, synth_text=None, synth_lang=None, text_processors=None):
    text_norm = []
    if prompt_text:
        prompt_tp = text_processors[prompt_lang]
        prompt_text = clean_text(prompt_text)
        prompt_text_encoded = prompt_tp.encode_text(prompt_text)
        text_norm += prompt_text_encoded
        if not prompt_text.strip().endswith(('.', '!', '?')):
            text_norm += prompt_tp.encode_text('.')
        if synth_text:
            # add period to indicate end of prompt
            text_norm += prompt_tp.encode_text('.  ')
    
    if synth_text:
        synth_tp = text_processors[synth_lang]
        synth_text = clean_text(synth_text)
        if not synth_text.strip().endswith(('.', '!', '?')):
            synth_text += '.'
        synth_text_encoded = synth_tp.encode_text(synth_text)
        text_norm += synth_text_encoded

    text_norm = torch.IntTensor(text_norm)
    return text_norm


def get_text_processors(config_path):
    text_processors = {}
    with open(config_path, 'r') as fp:
        text_processors_dict = json.load(fp)

    for language, config in text_processors_dict.items():
        print(f"Loading text processor for {language}")
        text_processors[language] = TextProcessing(
            **config, language=language, add_bos_eos_to_text=True)
    return text_processors


def get_transcript(audio, language, model, processor, sr=22050, model_name='whisper'):
    if sr != 16000:
        audio_s2t = TAF.resample(audio, sr, 16000)
    else:
        audio_s2t = audio

    if model_name == 'whisper':
        input_features = processor(
            audio_s2t.cpu(), return_tensors="pt", sampling_rate=16000).input_features
        input_features = input_features.to(audio_s2t.dtype).to(audio.device)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe")
        predicted_ids = model.generate(input_features.cuda())
        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True)
        transcription = transcription[0].strip()
        return transcription
    elif model_name == 'hubert-large':
        input_values = processor(
            audio_s2t.cpu(), return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(audio_s2t.dtype).to(audio.device)
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        return transcription
