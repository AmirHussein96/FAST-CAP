import json
import argparse
import ndjson
import os
import shutil
import unicodedata
from unidecode import unidecode
import re

def write_ndjson(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        ndjson.dump(data, f, ensure_ascii=True)

QUOTE_CHARS = '"“”'

def remove_diacritics(manifest: str, transliterate_nonlatin: bool = True):
    
    
    count = 0
    data = []
    dict1 = {}
    dict2 = {}
    manifest_name = os.path.basename(manifest)
    tmp_output = f"/tmp/fixed_{manifest_name}"
    with open(manifest, "r", encoding="utf-8") as f1:
        for line1 in f1:
            sample1 = json.loads(line1)
            text = sample1["text"]
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            if transliterate_nonlatin:
                text = unidecode(text)
            nfkd_form = unicodedata.normalize('NFD', text)
            no_diacritics = ''.join(ch for ch in nfkd_form if unicodedata.category(ch) != 'Mn')
            no_diacritics = re.sub(r'(?:\\"|")(.*?)(?:\\"|")', r'\1', no_diacritics)
            no_diacritics = re.sub(r'["“”](.*?)["“”]', r'\1', no_diacritics)
            no_diacritics = no_diacritics.replace('"', '')
            text = text.translate({ord(c): None for c in QUOTE_CHARS})
            no_diacritics = re.sub(r'\s+', ' ', no_diacritics).strip()
            sample1["text"] = no_diacritics
            data.append(sample1)
    write_ndjson(data, tmp_output)
    shutil.move(tmp_output, manifest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for non-ASCII characters in manifest field")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the reference.jsonl manifest file")

    args = parser.parse_args()

    remove_diacritics(args.manifest)
