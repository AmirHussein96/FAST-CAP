import argparse
import csv
import json
import os
from pathlib import Path
import soundfile as sf

lang_map = {
    "de": "de-DE",
    "es": "es-US",
    "fr": "fr-FR",
    "en": "en-US",
}

def get_duration(wav_path: str):
    try:
        with sf.SoundFile(wav_path) as f:
            return round(len(f) / f.samplerate, 3)
    except Exception:
        return None


def read_origin_tsv(origin_tsv: str, lang: str):
    """
    Read original (source-language) TSV.
    Returns a dict keyed by audio_id.
    """
    data = {}

    with open(origin_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_id = Path(row["path"]).stem
            data[audio_id] = {
                "speaker": row["client_id"],
                "text": row["sentence"].strip(),
                "lang": lang,
            }

    return data


def read_translation_tsv(translated_tsv: str):
    """
    Read translated TSV.
    Assumes columns: audio_id <TAB> translated_text
    """
    translations = {}

    with open(translated_tsv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            audio_id = Path(row[0]).stem
            translations[audio_id] = row[1].strip()

    return translations


def main(args):
    source_lang, target_lang = args.lang.split("_")
    origin_data = read_origin_tsv(args.origin_tsv, args.lang)
    translated_data = read_translation_tsv(args.translated_tsv)
    output_file = os.path.join(args.output_dir, f"manifest_{args.lang}_{args.subset}.jsonl")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for audio_id, origin in origin_data.items():
            if audio_id not in translated_data:
                continue

            wav_path = os.path.join(
                args.source_audio_root,
                args.lang,
                "clips",
                f"{audio_id}.wav",
            )

            if not os.path.exists(wav_path):
                continue

            duration = get_duration(wav_path)
            if duration is None:
                continue

            manifest = {
                "audio_filepath": wav_path,
                "duration": duration,
                "text": origin["text"],
                "text_translated_clean": translated_data[audio_id],
                "speaker": origin["speaker"],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "taskname": "translate",
            }

            out_f.write(json.dumps(manifest, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert source + translated Common Voice TSVs to NeMo JSONL manifest"
    )

    parser.add_argument(
        "--origin_tsv",
        required=True,
        help="Path to source-language TSV (Common Voice format)",
    )

    parser.add_argument(
        "--translated_tsv",
        required=True,
        help="Path to TSV containing translations",
    )

    parser.add_argument(
        "--source-audio-root",
        required=True,
        help="Root directory containing converted WAV files",
    )

    parser.add_argument(
        "--lang",
        required=True,
        help="Language pair (e.g., de_en, es_en)",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSONL manifests",
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="Subset (e.g., dev, test, train)",
    )

    args = parser.parse_args()
    main(args)
