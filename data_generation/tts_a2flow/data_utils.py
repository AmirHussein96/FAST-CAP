import torch
from lhotse import CutSet
import lhotse.dataset
from bigvgan.meldataset import mel_spectrogram
from torch.nn.utils.rnn import pad_sequence
from eval_utils_multilingual import normalize_mel, get_text
import re
from unidecode import unidecode
import unicodedata


QUOTE_CHARS = '"“”'

def clean_en_text(text):
    text = unidecode(text)
    nfkd_form = unicodedata.normalize('NFD', text)
    no_diacritics = ''.join(ch for ch in nfkd_form if unicodedata.category(ch) != 'Mn')
    no_diacritics = re.sub(r'(?:\\"|")(.*?)(?:\\"|")', r'\1', no_diacritics)
    no_diacritics = re.sub(r'["“”](.*?)["“”]', r'\1', no_diacritics)
    no_diacritics = no_diacritics.replace('"', '')
    text = text.translate({ord(c): None for c in QUOTE_CHARS})
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text):
    text = re.sub(r'\s+([.,!?;:])', r'\1', text) # Remove spaces before punctuation "capital ." → "capital."
    text = re.sub(r'(?:\\"|")(.*?)(?:\\"|")', r'\1', text) # Remove quotes around a word and \"\"
    text = re.sub(r'["“”](.*?)["“”]', r'\1', text)
    text = text.replace('"', '')
    text = text.translate({ord(c): None for c in QUOTE_CHARS})
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_func(lang):
    if lang == 'en-US':
        return clean_en_text
    else:
        return clean_text


class ToAudio(torch.utils.data.Dataset):
    def __init__(self, text_processors, max_pause=1, max_segment_duration=5.0):
        self.max_pause = max_pause
        self.max_segment_duration = max_segment_duration
        self.text_processors = text_processors

    def __getitem__(self, cuts: CutSet):
        # Trim cuts to alignments and get the first segment as prompt
        # source_chars = [len(cut.supervisions[0].text.replace(" ", "")) for cut in cuts]
        # translated_chars = [len(cut.custom.get('text_translated_clean').replace(" ", "")) for cut in cuts]
        
        # Compute source characters per second
        dur = [cut.duration for cut in cuts]
        text = [self.normalize_ellipsis(clean_func(cut.custom['source_lang'])(cut.supervisions[0].text)) for cut in cuts]
        # cps_source = [ dur[i] / source_chars[i] for i in range(len(dur))]
        
        # Compute expected duration to speak translated text at source CPS
        

        prompts = cuts.trim_to_alignments(type="word",
                                            max_pause=self.max_pause,
                                            max_segment_duration=self.max_segment_duration,
                                            get_all_segments=False)
        prompts = prompts.sort_by_duration(ascending=False)
        prompt_dur = [cut.duration for cut in prompts]
        prompt_text = [self.normalize_ellipsis(clean_func(cut.custom['source_lang'])(cut.supervisions[0].text)) for cut in prompts]
        translated_text = [self.normalize_ellipsis(clean_func(cut.custom['target_lang'])(cut.custom['text_translated_clean'])) for cut in prompts]
        source_lang = ['es_MX' if cut.custom['source_lang'] == 'es-US' else cut.custom['source_lang'].replace('-', '_') for cut in prompts]
        target_lang = ['es_MX' if cut.custom['target_lang'] == 'es-US' else cut.custom['target_lang'].replace('-', '_') for cut in prompts]
        dur_factors, expected_duration = self.get_exp_duration(src_langs=source_lang,
                                                tgt_langs=target_lang,
                                                src_dur=dur, 
                                                prompt_dur=prompt_dur, 
                                                prompt_text=prompt_text, 
                                                translated_texts=translated_text)
        audios = prompts.load_audio(collate=False)
        mels = self.extract_features(audios)
        if len(mels) == 0:
  
            # Skip: empty mel list 
            print(f"skipped file with empty mel: {cuts[0].supervisions[0].id}")
            print(prompt_dur)
            print(audios)
            return None
        mel_lens = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
        mels = [normalize_mel(mel) for mel in mels]
        pad_mels = pad_sequence([mel.T for mel in mels], batch_first=True, padding_value=0).transpose(1, 2)

        
        
        id_ = [cut.supervisions[0].id for cut in prompts]
        return {"id": id_, "audio": audios, "text": prompt_text, "expected_duration": expected_duration, "translated_text": translated_text, "mels": pad_mels, "mel_lens": mel_lens, "source_lang": source_lang, "target_lang": target_lang, "dur_factors": dur_factors, "src_dur": dur}

    def extract_features(self, audios):
        # downsample_16k = torchaudio.transforms.Resample(22050, 16000)
        mels = []
        for wav in audios:
            wav = torch.FloatTensor(wav)
            if wav.max() <= 1.5 and wav.min() >= -1.5:
                wav = wav
            else:
                wav = wav + torch.rand_like(wav)
                wav = wav / 32768.0
            # wav = wav.unsqueeze(0)
            mel = mel_spectrogram(wav, 1024, 80, 22050, 256, 1024, 0., 11025, center=False).squeeze()
            mels.append(mel)
        return mels

    def get_exp_duration(self, src_langs, tgt_langs, src_dur, prompt_dur, prompt_text, translated_texts):
        src_phones = [get_text(
            prompt_text=src_txt, 
            prompt_lang=lang, 
            text_processors=self.text_processors
            ) for src_txt, lang in zip(prompt_text, src_langs)]
        tgt_phones = [get_text(
            prompt_text=tgt_txt, 
            prompt_lang=lang, 
            text_processors=self.text_processors
            ) for tgt_txt, lang in zip(translated_texts, tgt_langs)]
       

        pps_source = [prompt_dur[i] / len(src_phones[i]) for i in range(len(prompt_dur))] # phone dur 
        # print(f"source dur: {prompt_dur}")
        # print(f"ratio: {pps_source}")
        # print(f"len pps_source: {len(src_phones[0])}")
        # print(f"pps_source: {src_phones}")
        # print(f"len tgt_phones: {len(tgt_phones[0])}")
        # print(f"tgt_phones: {tgt_phones}")


        tgt_exp_dur = [len(tgt_phones[i]) * pps_source[i] for i in range(len(pps_source))]
        dur_factors = [
            (src_dur[i]/ tgt_exp_dur[i]) if (src_dur[i] / tgt_exp_dur[i]) > 1.05 else 1
            for i in range(len(pps_source))]
        #print(f"dur_factors: {dur_factors}")

        return dur_factors, tgt_exp_dur

    def normalize_ellipsis(self, text):
        # Replace unicode ellipsis with dot
        text = text.replace('…', '.')
        # Replace sequences of multiple dots (e.g. .., ..., ....) with a single dot
        text = re.sub(r'\.{2,}', '.', text)
        return text

def get_lhotse_dataloader(cuts, batch_size, text_processors, num_workers=1, max_pause=0.5, max_segment_duration=5.0):
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(max_pause=max_pause, max_segment_duration=max_segment_duration, text_processors=text_processors),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=batch_size),
        num_workers=num_workers,
        batch_size=None,
    )
    return dloader