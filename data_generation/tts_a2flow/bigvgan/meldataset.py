# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pathlib
import pandas as pd
import soundfile as sf
import torchaudio
from tqdm import tqdm
# import dac
from soundfile import LibsndfileError

MAX_WAV_VALUE = 32767.0 # NOTE: -1 of public model to prevent int16 overflow (results in popping sound on corner cases)

# temporary hard-coded filelist not found in cs-oci-ord shared data storage
# currently some files in /lustre/fsw/portfolios/adlr/projects/adlr_audio_speech/datasets/LibriLight/Portuguese/raw_mp3/librivox-2961 are missing
FILE_NOT_FOUND_LIST = [
    "librivox-2961/canaa_10_aranha_64kb.mp3",
    "librivox-2961/canaa_09_aranha_64kb.mp3",
    "librivox-2961/canaa_02_aranha_64kb.mp3",
    "librivox-2961/canaa_01_aranha_64kb.mp3",
    "librivox-2961/esauejaco_08_assis_64kb.mp3",
    "librivox-2961/paginasrecolhidas_03_assis_64kb.mp3",
    "librivox-2961/orpheuno1_16__64kb.mp3"
]

LIBRITTS_EXCLUDE_SPEAKER_LIST = [
    "train-clean-360/1259"
]

# old ver that only supports wav
# def load_wav(full_path, sr_target):
#     sampling_rate, data = read(full_path)
#     if sampling_rate != sr_target:
#         raise RuntimeError(
#             "Sampling rate of the file {} is {} Hz, but the model requires {} Hz".format(
#                 full_path, sampling_rate, sr_target
#             )
#         )
#     return data, sampling_rate


def load_random_wav_segment_to_torch(track, audiopath, segment_length, target_sampling_rate, stereo=False):
    target_segment_length = segment_length
    if track.samplerate != target_sampling_rate:
        segment_length = math.ceil(segment_length * (track.samplerate / target_sampling_rate))
    audio = None
    
    if track.frames < segment_length:
        track.seek(0)
        audio = track.read()
        audio = torch.from_numpy(audio).float()
        # drop too short audio below 8192, not useful for training anyway
        if (len(audio.shape) < 1) or (len(audio.shape) == 1 and audio.numel() < 8192) or (len(audio.shape) == 2 and audio.numel() < 8192 * audio.shape[1]): # corner case that fails to fetch segment_length
            print(f"WARNING: {audiopath} {audio.shape} from track.read() does not match shortest accepted length (8192)")
            return None
    else:
        start_idx = random.randint(0, track.frames - segment_length)
        track.seek(start_idx)
        audio = track.read(segment_length)
        audio = torch.from_numpy(audio).float()
        # at this point audio should be [segment_length], if anything goes wrong (corrupted source, etc), return None to re-roll index
        if (len(audio.shape) < 1) or (len(audio.shape) == 1 and audio.numel() != segment_length) or (len(audio.shape) == 2 and audio.numel() != segment_length * audio.shape[1]): # corner case that fails to fetch segment_length
            print(f"WARNING: {audiopath} {audio.shape} from track.read() does not match expected shape {segment_length}")
            return None
    
    if not stereo:
        if len(audio.shape) == 2: # Handle stereo or spatial sound
            if audio.shape[1] > 2:
                print("WARNING: audio is not mono or stereo with #channel {}. Will downmix to mono!: {}".format(audio.shape[1], track))
            audio = torch.mean(audio, dim=-1) # [frames]
        audio = audio.unsqueeze(0)  # Ensure shape is [1, frames]
    else:
        if len(audio.shape) == 2:
            if audio.shape[1] > 2:
                print("WARNING: audio is not stereo with #channel {}. Will downmix to stereo!: {}".format(audio.shape[1], track))
                left_channel = torch.mean(audio[:, :audio.shape[1]//2], dim=1)
                right_channel = torch.mean(audio[:, audio.shape[1]//2:], dim=1)
                audio = torch.stack([left_channel, right_channel], dim=0) # [2, frames]
        if len(audio.shape) == 1:
            audio = audio.repeat(2, 1)  # Duplicate mono track to make it "stereo" [2, frames]
        if audio.shape[0] != 2:
            audio = audio.permute(1, 0)  # Correctly order the dimensions to [2, frames]

    # Apply padding if necessary
    if audio.shape[-1] < segment_length:
        padding = (0, segment_length - audio.shape[-1])
        audio = torch.nn.functional.pad(audio, padding, "constant")

    # Resampling
    if track.samplerate != target_sampling_rate:
        # audio = torchaudio.functional.resample(audio, track.samplerate, target_sampling_rate, resampling_method="sinc_interp_kaiser")
        audio = torch.from_numpy(librosa.resample(audio.numpy(), orig_sr=track.samplerate, target_sr=target_sampling_rate))
        # trim last element to match target_segment_length (e.g., 16385 for 44khz downsampled to 24khz -> 16384)
        audio = audio[..., :target_segment_length]

    # Validation of audio shape
    if (not stereo and audio.shape != (1, target_segment_length)) or (stereo and audio.shape != (2, target_segment_length)):
        print(f"WARNING: {audiopath} {audio.shape} does not match expected shape [{('1' if not stereo else '2')}, {target_segment_length}]")
        return None

    # Skip silence audio (not useful signal)
    if abs(audio).max() < 1e-4:
        print(f"WARNING: segment is abs(audio).max() is {abs(audio).max()} < 1e-4, silent audio. Skipping {audiopath}")
        return None
    
    # Normalize audio if it's out of the range [-1, 1]
    if audio.max() > 1 or audio.min() < -1:
        audio = audio / (audio.abs().max() + 1e-5)
        
    return audio # [1, T] for mono, [2, T] for stereo

def load_full_wav_to_torch(track, audiopath, target_sampling_rate, stereo=False):
    track.seek(0)
    audio = track.read(track.frames)
    audio = torch.from_numpy(audio).float()
    
    if not stereo:
        if len(audio.shape) == 2: # Handle stereo or spatial sound
            if audio.shape[1] > 2:
                print("WARNING: audio is not mono or stereo with #channel {}. Will downmix to mono!: {}".format(audio.shape[1], track))
            # Downmix all channels to mono
            audio = torch.mean(audio, dim=-1) # [frames]
        audio = audio.unsqueeze(0)  # Ensure shape is [1, frames]
    else:
        # If stereo is required and there are more than two channels, downmix to two channels
        if len(audio.shape) == 2 and audio.shape[1] > 2:
            print("WARNING: audio is not stereo with #channel {}. Will downmix to stereo!: {}".format(audio.shape[1], track))
            left_channel = torch.mean(audio[:, :audio.shape[1]//2], dim=1)
            right_channel = torch.mean(audio[:, audio.shape[1]//2:], dim=1)
            audio = torch.stack([left_channel, right_channel], dim=0) # [2, frames]
        if len(audio.shape) == 1:
            audio = audio.repeat(2, 1)  # Duplicate mono track to make it "stereo" [2, frames]
        if audio.shape[0] != 2:
           audio = audio.permute(1, 0)  # Correctly order the dimensions to [2, frames]
            
    if track.samplerate != target_sampling_rate:
        # audio = torchaudio.functional.resample(audio, track.samplerate, target_sampling_rate, resampling_method="sinc_interp_kaiser")
        audio = torch.from_numpy(librosa.resample(audio.numpy(), orig_sr=track.samplerate, target_sr=target_sampling_rate))
            
    if not stereo:
        if audio.shape[0] != 1:
            raise AssertionError(f"{audiopath} {audio.shape} should be mono with shape [1, frames]")
    else:
        if audio.shape[0] != 2:
            raise AssertionError(f"{audiopath} {audio.shape} should be stereo with shape [2, frames]")
    
    # Normalize audio if it's out of the range [-1, 1]
    if audio.max() > 1 or audio.min() < -1:
        audio = audio / (audio.abs().max() + 1e-5)
    
    return audio # [1, T] for mono, [2, T] for stereo
    
    
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

# function that returns both linear spec and mel
def linear_and_mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    # y shape is [B, 1, T] for mono and [B, 2, T] for stereo
    if y.size(1) == 2: 
        # Compute spectrograms for each channel and concatenate them
        stereo_results = [
            linear_and_mel_spectrogram(
                y[:, i, :], n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center
            ) for i in range(2)
        ]
        # Concatenate results ([B, C_linear, T_frame], [B, C_mel, T_frame]) for each channel
        stereo_spec = torch.cat([result[0] for result in stereo_results], dim=1)
        stereo_mel = torch.cat([result[1] for result in stereo_results], dim=1)
        return stereo_spec, stereo_mel # [B, C_linear*2, T_frame], [B, C_mel*2, T_frame]
        
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y,
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    
    spec = spectral_normalize_torch(spec)
    mel = spectral_normalize_torch(mel)

    return spec, mel # [B(1), C_linear, T_frame], [B(1), C_mel, T_frame]

# a function version that only returns mel. so far been used by many repos so let's keep it
def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    # y shape is [B, 1, T] for mono and [B, 2, T] for stereo
    if y.size(1) == 2: 
        # Process each channel separately and concatenate the results for stereo
        results = [
            mel_spectrogram(y[:, i, :], n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center)
            for i in range(2)
        ]
        stereo_mel = torch.cat(results, dim=1)  # Concatenate along the channel dimension
        return stereo_mel  # [B, C_mel*2, T_frame]
    
    # if torch.min(y) < -1.0:
    #     print("min value is ", torch.min(y))
    # if torch.max(y) > 1.0:
    #     print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y,
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    mel = spectral_normalize_torch(mel)

    return mel # [B(1), C_mel, T_frame]


def calculate_uniform_repeats(dset_stats):
    total_samples = sum(stat['samples'] for stat in dset_stats.values())
    num_datasets = len(dset_stats)
    target_samples_per_dset = total_samples / num_datasets

    for dset_name, stats in dset_stats.items():
        current_samples = stats['samples']
        # Ensure current_samples is not zero to avoid division by zero
        if current_samples > 0:
            repeats_needed = max(0, int(target_samples_per_dset // current_samples) - 1)
            dset_stats[dset_name]['repeated'] = repeats_needed
        else:
            # Handle the case where current_samples is zero
            dset_stats[dset_name]['repeated'] = 0

    return dset_stats

    
# excerpts from https://gitlab-master.nvidia.com/ADLR/gangnamtts/-/blob/main/data.py?ref_type=heads#L291
def load_data(datasets, split="|", **kwargs):
    dataset = []
    n_hours = 0.0
    dset_stat_dict = {}
    # Define a template for each dataset's statistics with fixed field widths for headers and data
    header_template = "{:<30} {:>15} {:>15} {:>15} {:>20}"
    data_template = "{:<30} {:>15.2f} {:>15d} {:>15d} {:>20.2f}"
    
    for dset_name, dset_dict in datasets.items():
        n_max_files = dset_dict.get("n_max_files", -1)
        n_hours_dset = 0.0
        folder_path = dset_dict["basedir"]
        audio_dir = dset_dict["audio_dir"]
        filename = dset_dict["filelist"]
        n_repeats = dset_dict.get("n_repeats", 0)
        dummy_speaker = dset_dict.get("dummy_speaker", False)
        if n_max_files == 0:
            print(f"Skipping dataset {dset_name}, {dset_dict}")
            continue
        print(f"Loading dataset {dset_name}, {dset_dict}")
        language = dset_dict.get("language", None)
        assert os.path.exists(folder_path), f"{folder_path} not found!"
        wav_folder_prefix = os.path.join(folder_path, audio_dir)
        assert os.path.exists(wav_folder_prefix), f"{wav_folder_prefix} not found!"
        
        filelist_path = os.path.join(folder_path, filename)
        file_extension = os.path.splitext(filelist_path)[1]
        if file_extension == ".ndjson":
            dataset_cur, n_hours_dset = load_data_ndjson(
                filelist_path,
                split,
                wav_folder_prefix,
                dummy_speaker,
                language,
                n_max_files,
                **kwargs,
            )
        elif file_extension == ".txt":
            dataset_cur, n_hours_dset = load_data_filelist(
                filelist_path,
                split,
                wav_folder_prefix,
                dummy_speaker,
                language,
                n_max_files,
                **kwargs,
            )
        elif file_extension == ".csv":
            raise NotImplementedError
            delimiter = dset_dict.get("delimiter", "\|")
            dataset_cur, n_hours_dset = load_data_csv(
                filelist_path,
                split,
                wav_folder_prefix,
                dummy_speaker,
                language,
                n_max_files,
                delimiter,
            )

        n_hours += n_hours_dset
        print(f"{dset_name} has {n_hours_dset:.2f} hours, {len(dataset_cur)} samples, repeated {n_repeats}x")
        if len(dataset_cur):
            dataset_cur = dataset_cur * (n_repeats + 1)
            dataset.extend(dataset_cur)
        
        # Store each dataset's statistics using the data template
        dset_stat_dict[dset_name] = {
            "hours": n_hours_dset,
            "samples": len(dataset_cur),
            "repeated": n_repeats
        }
        
    print(f"\nDataset after filtering has {n_hours:.2f} hours, {len(dataset)} samples\n")
    print("Dataset summary, sorted in descending #Samples order")
    print(header_template.format("Dataset Name", "Hours", "Samples", "Repeated", "#Samples/#TotalSamples (%)"))
    dset_stat_dict = dict(sorted(dset_stat_dict.items(), key=lambda item: item[1]['samples'], reverse=True))
    for dset_name, stats in dset_stat_dict.items():
        # Calculate each dataset's share of total samples as a percentage
        effective_sample_share = (stats["samples"] / len(dataset)) * 100
        if stats["samples"] >= 0:
            print(data_template.format(dset_name, stats["hours"], stats["samples"], stats["repeated"], effective_sample_share))
        
    # dset_stat_dict = calculate_uniform_repeats(dset_stat_dict)
    # print("To make dataset sampling uniform:")
    # for dset_name, stats in dset_stat_dict.items():
    #     print(f"{dset_name}: Needs repeat {stats['repeated']} times")
    
    return dataset


def load_data_filelist(
    metadata_path,
    split,
    wav_folder_prefix,
    dummy_speaker,
    language,
    n_max_files,
    sampling_rate,
    dur_min,
    dur_max,
    pre_shuffle=False,
    debug=False # kwargs
):
    with open(metadata_path, encoding="utf-8") as f:
        data = [line.strip().split(split) for line in f]

    # pre-shuffle data: this will be useful for very large filelist that cannot reach full epoch for 4h job
    if pre_shuffle:
        random.shuffle(data)
    
    dataset_cur = []
    n_hours_dset = 0
    for j, d in enumerate(data):
        audiopath = os.path.join(wav_folder_prefix, d[0])
        
        # excluding some speakers within libritts
        if any(target in audiopath for target in LIBRITTS_EXCLUDE_SPEAKER_LIST):
            continue

        if language == "nonspeech": # only take audiopath and fill all others to dummy values
            text = "DUMMY"
            speaker = "DUMMY"
            emotion = "other"
            track = sf.SoundFile(audiopath)
            duration = duration = float(track.frames / track.samplerate)
        
        else: # speech data
            text = d[1] if len(d) > 1 else "DUMMY"
            # if len(text) < self.min_text_len or len(text) > self.max_text_len:
            #     continue
            speaker = "DUMMY" if (dummy_speaker or len(d) < 2) else d[2]
            if len(d) > 3:
                try:
                    emotion = d[3]
                    duration = float(d[4])
                except:
                    try:
                        emotion = "other"
                        # track = sf.SoundFile(audiopath)
                        # duration = float(track.frames / track.samplerate)
                        duration = 2. # just to pass the check of MSP-PODCAST-Publish-1.9
                    except:
                        print(f"WARNING: emotion & duration is corrupted in line {j}: {audiopath}")
                        continue
            else:
                emotion = "other"
                # track = sf.SoundFile(audiopath)
                # duration = float(track.frames / track.samplerate)
                duration = 2. # just to pass the check of MSP-PODCAST-Publish-1.9

        if duration < dur_min or duration > dur_max:
            continue
        
        # if debug: # try loading the data with soundfile. This is very slow
        #     try:
        #         track = sf.SoundFile(audiopath)
        #     except LibsndfileError:
        #         print(f'WARNING: failed to load {audiopath}')
            
        dataset_cur.append(
            {
                "audiopath": audiopath,
                "text": text,
                "speaker": speaker,
                "emotion": emotion,
                "duration": duration,
                "language": language,
            }
        )

        n_hours_dset += duration / 3600.0
        if n_max_files > -1 and len(dataset_cur) == n_max_files:
            break

    return dataset_cur, n_hours_dset


def load_data_ndjson(
    metadata_path,
    split,
    wav_folder_prefix,
    dummy_speaker,
    language,
    n_max_files,
    sampling_rate,
    dur_min,
    dur_max,
    pre_shuffle=False,
    debug=False # kwargs
):
    dataset_cur = []
    n_hours_dataset = 0.0
    total_rows_sampled = 0  # Track total rows sampled across chunks

    reader = pd.read_json(metadata_path, lines=True, chunksize=100000)  # to save memory

    for df in reader:
        if n_max_files != -1 and len(dataset_cur) >= n_max_files:
            break
        if "audio_filepath" in df:
            if "audiopath" in df:
                del df["audiopath"]
            df.rename(columns={"audio_filepath": "audiopath"}, inplace=True)
            
        # Pre-shuffle each chunk if requested
        if pre_shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        # keep only required,  decrease memory footprint
        columns_req = [
            "audiopath",
            "text",
            "speaker",
            "duration",
            "language",
            "start_time",
            "speech_rate",
        ]
        columns = [x for x in df.columns if x in columns_req]
        df = df[columns]
        
        # # filter given text len
        # df = df.loc[df['text'].str.len() >= self.min_text_len]
        # df = df.loc[df['text'].str.len() < self.max_text_len]

        # filter given audio duration
        df = df.loc[df["duration"] >= dur_min]
        df = df.loc[df["duration"] < dur_max]
        
        # TEMPORARY: filter out hard-coded banlist not in the cluster yet
        df = df.loc[~df["audiopath"].isin(FILE_NOT_FOUND_LIST)]

        # # drop sentences with numbers
        # if self.drop_sentences_with_number:
        #     df = df[~df.text.str.contains(r'\d')]

        # # filter given speech rate (graphemes per second)
        # if 'speech_rate' in df:
        #     df = df.loc[df['speech_rate'] >= self.speech_rate_min]
        #     df = df.loc[df['speech_rate'] < self.speech_rate_max]



        df["audiopath"] = df["audiopath"].apply(
            lambda x: os.path.join(wav_folder_prefix, x)
        )
        
        # TODO: generalize this
        if "NVYT-Speech" in metadata_path:
            # Filter out rows where the 'audiopath' file does not exist
            df = df[df['audiopath'].apply(os.path.exists)]
        
        df["speaker"] = "DUMMY" if dummy_speaker else df["speaker"]
        df["language"] = language

        # now that samples have been filtered, sample n_max_files
        if n_max_files != -1:
            # Adjust how many more rows we can sample
            rows_left_to_sample = n_max_files - total_rows_sampled
            if rows_left_to_sample <= 0:
                break  # Stop if we've already sampled enough rows
            sample_size = min(len(df), rows_left_to_sample)
            df = df.sample(n=sample_size).reset_index(drop=True)
            total_rows_sampled += sample_size
            
        # if debug: # try loading the data with soundfile
        #     for index, row in tqdm(df.iterrows()):
        #         audiopath = row["audiopath"]
        #         try:
        #             track = sf.SoundFile(audiopath)
        #         except LibsndfileError:
        #             print(f'WARNING: failed to load {audiopath}')
        n_hours_dataset += df["duration"].sum() / 3600
        dataset_cur.extend(df.to_dict("records"))
    return dataset_cur, n_hours_dataset


# old one from public repo
def get_dataset_filelist(a):
    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        training_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0])
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print("first training file: {}".format(training_files[0]))

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        validation_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0])
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print("first validation file: {}".format(validation_files[0]))

    list_unseen_validation_files = []
    for i in range(len(a.list_input_unseen_validation_file)):
        with open(a.list_input_unseen_validation_file[i], "r", encoding="utf-8") as fi:
            unseen_validation_files = [
                os.path.join(a.list_input_unseen_wavs_dir[i], x.split("|")[0])
                for x in fi.read().split("\n")
                if len(x) > 0
            ]
            print(
                "first unseen {}th validation fileset: {}".format(
                    i, unseen_validation_files[0]
                )
            )
            list_unseen_validation_files.append(unseen_validation_files)

    return training_files, validation_files, list_unseen_validation_files

class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        hparams,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=True,
        debug=False
    ):
        self.audio_files = [file["audiopath"] for file in training_files]
        self.hparams = hparams
        self.is_seen = is_seen
        # self.name = '-'.join(pathlib.Path(self.audio_files[0]).parts[1:4]).strip("/")
        self.name = "DATASET"  # placeholder
        self.segment_size = self.hparams.segment_size
        self.sampling_rate = self.hparams.sampling_rate
        self.split = split
        self.n_fft = self.hparams.n_fft
        self.num_mels = self.hparams.num_mels
        self.hop_size = self.hparams.hop_size
        self.win_size = self.hparams.win_size
        self.fmin = self.hparams.fmin
        self.fmax = self.hparams.fmax
        self.fmax_for_loss = self.hparams.fmax_for_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        # special flag whether to normalize volume or not
        # if not found in config, default is set to true to be same as original hifi-gan & bigvgan
        # but if "normalize_volume=false" is explicitly set in config, it will be switched off
        self.normalize_volume = getattr(self.hparams, "normalize_volume", True)
        if not self.normalize_volume:
            print("WARNING: normalize_volume is set to False. waveform will NOT be volume-normalized during training!")
        self.stereo = getattr(self.hparams, "stereo", False)
        if self.stereo:
            print("WARNING: stereo is set to True. Make sure that the model is stereo supported!")

        # # comment out these lines when full sanity check before main jobs is complete
        # if debug:
        #     print("INFO: debug mode is on. checking dataset integrity...")
        #     indices_to_remove = []
        #     for i in tqdm(range(len(self.audio_files))):
        #         if not os.path.exists(self.audio_files[i]):
        #             print("WARNING: file not found {}".format(self.audio_files[i]))
        #             indices_to_remove.append(i)
        #     if indices_to_remove == []:
        #         print("INFO: all audio_files exist in the system!")
        #     else:
        #         print("INFO: filtering out not found audio_files...")
        #         for index in sorted(indices_to_remove, reverse=True):
        #             del self.audio_files[index]
        
        if shuffle:
            random.shuffle(self.audio_files)

    def __getitem__(self, index):
        if not self.fine_tuning:
            if self.split:
                audio = None
                while audio is None:
                    try:
                        audiopath = self.audio_files[index]
                        track = sf.SoundFile(audiopath)
                        if not track.seekable(): # corner case, re-roll
                            index = random.randint(0, len(self.audio_files))
                            continue
                        if track.frames == 0: # corner case, re-roll
                            index = random.randint(0, len(self.audio_files))
                            continue
                        audio = load_random_wav_segment_to_torch(track, audiopath, self.segment_size, self.sampling_rate, self.stereo)
                    except LibsndfileError: # corner case that fails to load.
                        print("WARNING: LibsndfileError {}".format(self.audio_files[index]))
                        index = random.randint(0, len(self.audio_files))
                        continue
                    except ValueError: # also mysterious corner case
                        print("WARNING: ValueError {}".format(self.audio_files[index]))
                        index = random.randint(0, len(self.audio_files))
                        continue
                    if audio is None:
                        print("WARNING: skipping suspicious audio {}".format(self.audio_files[index]))
                        index = random.randint(0, len(self.audio_files))
                    
                assert audio.shape[1] == self.segment_size,\
                    "audio shape {} does not match segment size {}: track {}".format(audio.shape, self.segment_size, track)
                if self.normalize_volume:
                    audio = audio / (audio.abs().max() + 1e-5) * 0.95 # L-inf volume normalization as in public hifi-gan & bigvgan
                spec, mel = linear_and_mel_spectrogram(
                    audio.unsqueeze(0), # [B(1), C, T]
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
                
            else:  # validation step
                try:
                    audiopath = self.audio_files[index]
                    track = sf.SoundFile(audiopath)
                    audio = load_full_wav_to_torch(track, audiopath, self.sampling_rate, self.stereo)
                except LibsndfileError:
                    print("ERROR: LibsndfileError for validation file {}".format(self.audio_files[index]))
                    print("validation file seems very wrong. check again!")
                    exit()
                # match audio length to self.hop_size * n for evaluation
                if (audio.shape[1] % self.hop_size) != 0:
                    audio = audio[:, : -(audio.shape[1] % self.hop_size)]
                if self.normalize_volume:
                    audio = audio / (audio.abs().max() + 1e-5) * 0.95 # L-inf volume normalization as in public hifi-gan & bigvgan
                spec, mel = linear_and_mel_spectrogram(
                    audio.unsqueeze(0), # [B(1), C, T]
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
            assert audio.shape[1] == (mel.shape[2] * self.hop_size), "audio shape {} mel shape {}".format(audio.shape, mel.shape)

        else:
            mel = np.load(
                os.path.join(
                    self.base_mels_path,
                    os.path.splitext(os.path.split(audiopath)[-1])[0] + ".npy",
                )
            )
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start
                        * self.hop_size : (mel_start + frames_per_seg)
                        * self.hop_size,
                    ]
                else:
                    mel = torch.nn.functional.pad(
                        mel, (0, frames_per_seg - mel.size(2)), "constant"
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

        mel_loss = mel_spectrogram(
            audio.unsqueeze(0), # [B(1), C, T]
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_for_loss,
            center=False,
        )

        # make it [C, T] for all features
        return_dict = {
            "linear_spec": spec.squeeze(),
            "mel": mel.squeeze(),
            "audio": audio,
            "audiopath": audiopath,
            "mel_loss": mel_loss.squeeze()
        }
        return return_dict
        #return (mel.squeeze(), audio.squeeze(0), audiopath, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
