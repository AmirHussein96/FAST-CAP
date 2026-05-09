import os
import json
import argparse
from collections import defaultdict
import numpy as np
import librosa
from scipy.io.wavfile import write

import torch
# from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from eval_utils import get_text_processors, get_text
from eval_utils import normalize_mel, denormalize_mel
from eval_utils import get_duration
from utils import set_hparams, HParams, load_a2flow, load_dp, load_vocoder

from bigvgan.meldataset import mel_spectrogram


class DataTTS(torch.utils.data.Dataset):
    def __init__(self, manifest_path, sampling_rate=22050, language='en_us'):
        self.data = self.load_ndjson(manifest_path)
        self.sampling_rate = sampling_rate
        self.language = language

    def get_mel(self, audiopath):
        audio = librosa.core.load(audiopath, sr=22050)[0]
        audio = torch.FloatTensor(audio)[None]
        mel = mel_spectrogram(
            audio, 1024, 80, 22050, 256, 1024, 0., 11025, center=False)
        return mel

    def load_ndjson(self, file_path, sort=True):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def collate(self, batch):
        batch = list(filter(lambda s: s is not None, batch))
        o_dict = defaultdict(list)
        for i, item in enumerate(batch):
            for key, value in item.items():
                o_dict[key].append(value)

        for key, value in o_dict.items():
            if 'lens' in key:
                o_dict[key] = torch.LongTensor(value)
            elif 'audio' in key:
                o_dict[key] = pad_sequence(value, batch_first=True)[:, None]
            else:
                continue

        return o_dict

    def __getitem__(self, index):
        row = self.data[index]
        row['prompt_mel'] = self.get_mel(row['prompt_path'])
        row['language'] = row.get('language', self.language)
        return row

    def __len__(self):
        return len(self.data)


def main(manifest_path, output_dir, generator_path, dp_path, vocoder_path,
         vocoder_config_path, n_timesteps, gradient_scale, alpha, dur_scale,
         sampling_rate, language, use_dp):
    args = HParams()
    args.generator_path = generator_path
    hps_generator = set_hparams(args.generator_path)

    # dp for English
    args.dp_path = dp_path
    hps_dp = set_hparams(args.dp_path)

    text_processors = get_text_processors(hps_generator.data.text_processors_config_path)
    generator = load_a2flow(args.generator_path, hps_generator, n_language=len(text_processors)).cuda().eval()
    dp = load_dp(args.dp_path, hps_dp, n_language=len(text_processors)).cuda().eval()
    vocoder = load_vocoder(vocoder_config_path, vocoder_path).cuda()

    # define model and inference hyperparams
    args.language = 'en_US'
    args.timesteps = n_timesteps
    args.gradient_scale = gradient_scale
    args.alpha = alpha
    args.dur_scale = dur_scale

    data = DataTTS(manifest_path, sampling_rate, language)

    os.makedirs(output_dir, exist_ok=True)
    for idx, row in enumerate(data):
        print(f"Processing row {idx+1} of {len(data)}")
        prompt_path = row['prompt_path']
        prompt_mel = row['prompt_mel']
        prompt_transcript = row['prompt_transcript']
        text = row['text']

        # prepare mel input
        p = prompt_mel.cuda()
        p_lengths = torch.LongTensor([p.shape[-1]]).cuda()
        p = normalize_mel(p)

        # prepare text input
        transcript = prompt_transcript
        text_gen = text
        text = transcript + '  ' + text_gen

        # compute sequence length
        text_dp_norm = get_text(text_gen, language=args.language, text_processors=text_processors)
        text_norm = get_text(text, language=args.language, text_processors=text_processors)
        x_dp = text_dp_norm.unsqueeze(0).cuda()
        x_dp_lengths = torch.LongTensor([x_dp.shape[-1]]).cuda()
        x = text_norm.unsqueeze(0).cuda()
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
        language_id = torch.LongTensor([list(text_processors.keys()).index(args.language)]).cuda()

        # scale length given dur scale
        if use_dp:
            y_lengths_dp = dp(x_dp, x_dp_lengths, p, p_lengths, language_id=language_id)
        else:
            y_lengths_dp = (p_lengths / (x_lengths - x_dp.size(-1) - 1) * (x_dp.size(-1) + 1)).long()
        dur_scale = max(min(dur_scale, 1.2), 0.8)
        y_lengths = (y_lengths_dp * dur_scale).long()

        # synthesize given sentence lengths and speaker prompt
        _, y_dec = generator.infer(
            x, x_lengths, p=p, p_lengths=p_lengths, y_lengths=y_lengths,
            n_timesteps=args.timesteps, gradient_scale=args.gradient_scale,
            language_id=language_id, alpha=args.alpha, texts=[text]
        )

        # denormalize mel and synthesize with vocoder
        y_dec = denormalize_mel(y_dec)
        audio = vocoder.forward(y_dec)[0, 0].cpu().numpy()

        # scale audio if it has values larger than 1.0
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 1.0:
            audio = audio / max_amplitude
        audio = (audio * 32767).astype(np.int16)

        filename, fileext = os.path.splitext(os.path.basename(prompt_path))
        audiopath = os.path.join(output_dir, f"{idx}_{filename}.wav")
        write(audiopath, sampling_rate, audio)

        # save entry to manifest
        result = {
            'audio_path': audiopath,
            'text': text,
            'duration': get_duration(audio.shape[0], sampling_rate)
        }
        with open(f"{output_dir}/manfiest.ndjson", "a", encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument('--generator_path')
    parser.add_argument('--dp_path')
    parser.add_argument('--vocoder_path')
    parser.add_argument('--vocoder_config_path')
    parser.add_argument('--n_timesteps', type=int, default=32)
    parser.add_argument('--gradient_scale', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--dur_scale', type=float, default=1.0)
    parser.add_argument('--sampling_rate', type=float, default=22050)
    parser.add_argument('--language', type=str, default='en_us')
    parser.add_argument('--use_dp', action='store_true')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    with torch.no_grad():
        main(args.manifest_path, args.output_dir, args.generator_path,
             args.dp_path, args.vocoder_path, args.vocoder_config_path,
             args.n_timesteps, args.gradient_scale, args.alpha, args.dur_scale,
             args.sampling_rate, args.language, args.use_dp)
