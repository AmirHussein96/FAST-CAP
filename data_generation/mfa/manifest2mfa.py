import json
import argparse
import os
import shutil
import re
import unicodedata
from unidecode import unidecode
import ndjson
import string

QUOTE_CHARS = '"“”'


def write_ndjson(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        ndjson.dump(data, f, ensure_ascii=True)

        
def clean_text(text):
    text = re.sub(r'\s+([.,!?;:¡¿])', r'\1', text) # Remove spaces before punctuation "capital ." → "capital."
    text = re.sub(r'(?:\\"|")(.*?)(?:\\"|")', r'\1', text)
    text = re.sub(r'["“”](.*?)["“”]', r'\1', text)
    text = text.replace('"', '')
    text = text.translate({ord(c): None for c in QUOTE_CHARS})
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_en_text(text):
    text = re.sub(r'\s+([.,!?;:])', r'\1', text) # Remove spaces before punctuation "capital ." → "capital."
    text = unidecode(text) # ASCII-fy (strip accents & non-ASCII look alikes) # "jubilación" → "jubilacion"
    nfkd_form = unicodedata.normalize('NFD', text)
    no_diacritics = ''.join(ch for ch in nfkd_form if unicodedata.category(ch) != 'Mn')
    text = re.sub(r'(?:\\"|")(.*?)(?:\\"|")', r'\1', text)
    text = re.sub(r'["“”](.*?)["“”]', r'\1', text)
    text = text.replace('"', '')
    text = text.translate({ord(c): None for c in QUOTE_CHARS})
    text = re.sub(r'([!?.,;:])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main(manifest: str, output_dir:str, txt_field: str = "text", lang: str = "en"):

    data = []
    count = 0
    audios_to_rm = set()
    manifest_name = os.path.basename(manifest)
    # dir_path = os.path.dirname(manifest)
    # with open(manifest, "r", encoding="utf-8") as f1, open(f"{dir_path}/list.txt", "w", encoding="utf-8") as f2:
    with open(manifest, "r", encoding="utf-8") as f1:
        for line1 in f1:
            sample1 = json.loads(line1)
            # if sample1["_skipme"] == 0:
            count += 1
            name_without_ext = os.path.splitext(os.path.basename(sample1["audio_filepath"]))[0]
            sample_id = sample1["audio_filepath"]
            data.append(sample1)
            txt = sample1["text"]
            if lang == "en":
                txt = clean_en_text(txt)
            else:
                txt = clean_text(txt)
            # f2.write(f"{sample_id}\t{txt}\n")
            out_file = f"{name_without_ext}.txt"
            with open(os.path.join(output_dir, out_file), 'w', encoding="utf-8") as fout:
                fout.write(txt)
            # else:
            #     audios_to_rm.add(sample_id)
    # for audio in audios_to_rm:
    #     os.remove(os.path.join(output_dir, audio))
    # write_ndjson(data, os.path.join(output_dir, manifest_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert generate MFA files from Nemo manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the reference.jsonl manifest file")
    parser.add_argument("--txt-field", type=str, default="text", help="Field to check (default: 'text')")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--lang", type=str)

    args = parser.parse_args()

    main(args.manifest, txt_field=args.txt_field, output_dir=args.output_dir, lang=args.lang)
