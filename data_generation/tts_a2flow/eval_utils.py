import json
import librosa
from librosa import resample as librosa_resample
import numpy as np
import torch
import torchaudio
import torchaudio.functional as TAF
from tts_text_processing.text_processing import TextProcessing
import datetime as dt
import IPython.display as ipd
from torch.cuda import amp
from bigvgan.meldataset import mel_spectrogram
from transformers import Wav2Vec2Processor, HubertForCTC, AutoProcessor, AutoModelForSpeechSeq2Seq


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


def get_item_vc(wav, wav_gen, mel, mel_gen):
    return [{'wav': wav, 'wav_gen': wav_gen, 'p': mel, 'p_gen': mel_gen}]


def get_s2t(use_whisper):
    if use_whisper:
        s2t_processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
        s2t = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
        print("Load Whisper to transcribe the prompt")
    else:
        s2t_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        s2t = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        print("Load HuBERT-L to transcribe the prompt")
    return s2t, s2t_processor


def get_prompt(sample_path, num_second=10000, transcript=None, use_whisper=True, tts=True):
    downsample_16k = torchaudio.transforms.Resample(22050, 16000)
    if tts:
        s2t, s2t_processor = get_s2t(use_whisper)
        s2t = s2t.eval().cuda()

    wav, _ = librosa.load(sample_path, sr=22050)
    wav = torch.FloatTensor(wav)[:int(num_second * 22050)]
    print("Load Audio")
    ipd.display(ipd.Audio(wav, rate=22050))
    if wav.max() <= 1.5 and wav.min() >= -1.5:
        wav = wav
    else:
        wav = wav + torch.rand_like(wav)
        wav = wav / 32768.0
    wav = wav.unsqueeze(0)
    mel = mel_spectrogram(wav, 1024, 80, 22050, 256, 1024, 0., 11025, center=False).squeeze()
    wav_16k = downsample_16k(wav)
    if transcript is None and tts:
        # if transcript is None, use ASR model to extract transcript from the prompt
        wav_16k_transcript = torch.cat((torch.zeros((1, 400)), wav_16k, torch.zeros((1, 400))), dim=1)
        if use_whisper:
            transcript = get_transcript(wav_16k_transcript.squeeze(0).cuda(), 'english', s2t, s2t_processor, sr=16000, model_name='whisper')
        else:
            transcript = get_transcript(wav_16k_transcript.squeeze(0).cuda(), 'english', s2t, s2t_processor, sr=16000, model_name='hubert-large')
        print(f"Transcript of the prompt: {transcript}")
    return wav, mel, transcript


@torch.no_grad()
def sample(args, item_list, text_processors, generator, dp, vocoder):
    audio_list = []
    for i, item in enumerate(item_list):
        transcript = item['transcript']
        text_gen = item['text']
        text = transcript + '  ' + text_gen

        wav_gt = item['wav'].unsqueeze(0).cuda()

        transcript_norm = get_text(transcript, language=args.language, text_processors=text_processors)
        text_dp_norm = get_text(text_gen, language=args.language, text_processors=text_processors)
        text_norm = get_text(text, language=args.language, text_processors=text_processors)
        x_p = transcript_norm.unsqueeze(0).cuda()
        x_p_lengths = torch.LongTensor([x_p.shape[-1]]).cuda()
        x_dp = text_dp_norm.unsqueeze(0).cuda()
        x_dp_lengths = torch.LongTensor([x_dp.shape[-1]]).cuda()
        x = text_norm.unsqueeze(0).cuda()
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
        language_id = torch.LongTensor([list(text_processors.keys()).index(args.language)]).cuda()
        p = item['p'].unsqueeze(0).cuda()
        p_lengths = torch.LongTensor([p.shape[-1]]).cuda()
        p = normalize_mel(p)

        print(f"Transcript of Prompt: {transcript}")
        print("Prompt Audio")
        ipd.display(ipd.Audio(item['wav'], rate=22050))

        torch.cuda.synchronize()
        t = dt.datetime.now()
        with amp.autocast(enabled=args.fp16):
            if getattr(args, "use_dp", True):
                y_lengths_dp = dp(x_dp, x_dp_lengths, p, p_lengths, language_id=language_id)
            else:
                y_lengths_dp = (p_lengths / (x_lengths - x_dp.size(-1) - 1) * (x_dp.size(-1) + 1)).long()
            if not args.use_prompt:
                x, x_lengths = x_dp, x_dp_lengths
                p, p_lengths = None, None
            dur_scale = getattr(args, "dur_scale", 1)
            dur_scale = max(min(dur_scale, 1.2), 0.8)
            y_lengths = (y_lengths_dp * dur_scale).long()
            _, y_dec = generator.infer(
                x, x_lengths, p=p, p_lengths=p_lengths, y_lengths=y_lengths, n_timesteps=args.timesteps,
                gradient_scale=args.gradient_scale, language_id=language_id, alpha=args.alpha, texts=[text]
            )
        y_dec = denormalize_mel(y_dec)
        torch.cuda.synchronize()
        t1 = (dt.datetime.now() - t).total_seconds()

        print(f"Text input: {text_gen}")
        print("Generated Audio")
        with torch.no_grad():
            audio = vocoder.forward(y_dec.detach())
            audio = (audio.cpu().squeeze(1).clamp(-1, 1).numpy() * 32768).astype(np.int16)
            audio_list.append(audio)
            ipd.display(ipd.Audio(audio, rate=22050))
        torch.cuda.synchronize()
        t2 = (dt.datetime.now() - t).total_seconds()
        print(f'A2Flow RTF: {t1 * 22050 / audio.shape[-1]}')
        print(f'BigVGAN RTF: {(t2-t1) * 22050 / audio.shape[-1]}')
        print(f'Total RTF: {t2 * 22050 / audio.shape[-1]}')
        print(f'Total Time: {t1}sec')
        print('===============================================================')
    return audio_list


@torch.no_grad()
def sample_vc(args, item_list, text_processors, generator, vocoder):
    audio_list = []
    for i, item in enumerate(item_list):
        wav_gt = item['wav']
        wav_gen = item['wav_gen']
        wav_all = torch.cat((wav_gt, wav_gen), dim=-1)
        wav_all_np = wav_all.squeeze(0).detach().numpy()
        wav_16k = librosa_resample(wav_all_np, orig_sr=22050, target_sr=16000)
        wav_16k = torch.tensor(wav_16k).unsqueeze(0)
        wav_gt, wav_16k = wav_gt.cuda(), wav_16k.cuda()
        wav_16k_lengths = torch.LongTensor([wav_16k.size(-1)]).cuda()
        
        language_id = torch.LongTensor([list(text_processors.keys()).index(args.language)]).cuda()
        p = item['p'].unsqueeze(0).cuda()
        p_gen = item['p_gen']
        p_lengths = torch.LongTensor([p.shape[-1]]).cuda()
        y_lengths = torch.LongTensor([p_gen.shape[-1]]).cuda()
        p = normalize_mel(p)

        print("Prompt Audio")
        ipd.display(ipd.Audio(item['wav'], rate=22050))

        torch.cuda.synchronize()
        t = dt.datetime.now()
        with amp.autocast(enabled=args.fp16):
            if not args.use_prompt:
                p, p_lengths = None, None
            dur_scale = getattr(args, "dur_scale", 1)
            dur_scale = max(min(dur_scale, 1.2), 0.8)
            y_lengths = (y_lengths * dur_scale).long()
            _, y_dec = generator.infer(
                p=p, p_lengths=p_lengths, y_lengths=y_lengths, n_timesteps=args.timesteps,
                gradient_scale=args.gradient_scale, language_id=language_id, alpha=args.alpha,
                audio_16k=wav_16k, audio_16k_lengths=wav_16k_lengths, use_unit=True
            )
        y_dec = denormalize_mel(y_dec)
        torch.cuda.synchronize()
        t1 = (dt.datetime.now() - t).total_seconds()

        print("Generated Audio")
        with torch.no_grad():
            audio = vocoder.forward(y_dec.detach())
            audio = (audio.cpu().squeeze(1).clamp(-1, 1).numpy() * 32768).astype(np.int16)
            audio_list.append(audio)
            ipd.display(ipd.Audio(audio, rate=22050))
        torch.cuda.synchronize()
        t2 = (dt.datetime.now() - t).total_seconds()
        print(f'A2Flow RTF: {t1 * 22050 / audio.shape[-1]}')
        print(f'BigVGAN RTF: {(t2-t1) * 22050 / audio.shape[-1]}')
        print(f'Total RTF: {t2 * 22050 / audio.shape[-1]}')
        print(f'Total Time: {t1}sec')
        print('===============================================================')
    return audio_list
        

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


def get_text(text, language, text_processors):
    # sentence -> phoneme index sequence
    tp = text_processors[language]
    text_norm = tp.encode_text(text)
    text_norm = torch.IntTensor(text_norm)
    return text_norm


def get_text_processors(config_path):
    text_processors = {}
    with open(config_path, 'r') as fp:
        text_processors_dict = json.load(fp)

    for language, config in text_processors_dict.items():
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
