#!/usr/bin/env python3
import json
import argparse
import os
from pathlib import Path
import re
import unicodedata
from unidecode import unidecode

"""
This script extracts parallel text pairs from NeMo-style manifests.

It expects each entry in the manifest to contain the fields:
- "text": the source text
- "text_translated_clean": the corresponding cleaned translation
"""

QUOTE_CHARS = '"“”'

def strip_weird_acute(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Replace spacing acute (U+00B4) with ASCII apostrophe
    s = s.replace("\u00B4", "'")
    # Replace combining acute (U+0301) with ASCII apostrophe if it appears standalone after a space, else drop it
    s = s.replace("\u0301", "'")
    s = s.replace("\u2019", "'")
    return s

def normalize_ellipsis(text):
        # Replace unicode ellipsis with dot
        text = text.replace('…', '.')
        # Replace sequences of multiple dots (e.g. .., ..., ....) with a single dot
        text = re.sub(r'\.{2,}', '.', text)
        return text

def clean_tgt_text(text: str) -> str:
    text = strip_weird_acute(text)
    text = normalize_ellipsis(text)

    # Target side: keep your current normalization (transliterate + strip diacritics)
    text = unidecode(text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    # Unwrap quoted content
    text = re.sub(r'\\"(.*?)\\"', r'\1', text)
    text = re.sub(r'["“”](.*?)["“”]', r'\1', text)
    # Remove any stray quotes
    text = text.translate({ord(c): None for c in QUOTE_CHARS})
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text: str) -> str:

    # Unwrap quoted content
    text = re.sub(r'\\"(.*?)\\"', r'\1', text)
    text = re.sub(r'["“”](.*?)["“”]', r'\1', text)
    # Remove any stray quotes
    text = text.translate({ord(c): None for c in QUOTE_CHARS})
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_manifest(manifest_path, output_dir, clean_tgt):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames based on manifest name
    manifest_name = Path(manifest_path).stem
    parallel_file = os.path.join(output_dir, f"{manifest_name}.parallel")
    ids_file = os.path.join(output_dir, f"{manifest_name}.ids")
    
    with open(manifest_path, 'r', encoding='utf-8') as f_in, \
         open(parallel_file, 'w', encoding='utf-8') as f_out, \
         open(ids_file, 'w', encoding='utf-8') as f_ids:
        
        for line in f_in:
            try:
                data = json.loads(line.strip())
                source_text = clean_text(data.get('text', '').strip())
                source_text = normalize_ellipsis(source_text)
                if clean_tgt:
                    translation = clean_tgt_text(data.get('text_translated_clean', '').strip())
                else:
                    translation = data.get('text_translated_clean', '').strip()
                audio_id = data.get('audio_filepath', '').strip()
                
                if source_text and translation:  # Only write if both fields exist
                    f_out.write(f"{source_text} ||| {translation}\n")
                    f_ids.write(f"{audio_id}\n")
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {manifest_path}")
                continue
            except Exception as e:
                print(f"Error processing line in {manifest_path}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Process manifest files to extract parallel text and IDs')
    parser.add_argument('manifest_path', help='Path to the manifest file')
    parser.add_argument('--output-dir', default='manifest_output', help='Output directory for parallel and IDs files')
    parser.add_argument('--clean-tgt', action='store_true', help='Clean target text before alignment')
    
    args = parser.parse_args()
    process_manifest(args.manifest_path, args.output_dir, args.clean_tgt)

if __name__ == '__main__':
    main() 

    