"""
Evaluate Speech Translation (ST) predictions from JSON or JSONL output files.

This script computes the standard MT/ST metrics:
    - BLEU  (SacreBLEU implementation)
    - chrF++ (SacreBLEU implementation)
    - COMET  (optional, requires `unbabel-comet`)

It expects a file containing JSON objects with at least the following fields:
    {
        "target_text": "reference translation",
        "pred_text": "text output from ST model",
        "speech_pred_transcribed": "ASR→MT transcription output (optional)"
    }

The file can be either:
    • JSONL (one JSON object per line)
    • concatenated JSON ("}{") format

Each prediction field is evaluated independently against `target_text`.

------------------------------------------------------------
USAGE
------------------------------------------------------------

Basic:
    python score_st.py --json results.jsonl

Compute BLEU and chrF++ (default). Prints metrics for:
    - `pred_text`
    - `speech_pred_transcribed` (if present)

Optional arguments:
    --use-comet                 # Compute COMET score in addition
    --comet-ckpt PATH           # Path to a COMET checkpoint (.ckpt)
    --lower                     # Lowercase text before scoring
    --rm-punctuation            # Remove punctuation before scoring
    --clean-english-text        # Clean and normalize English text
    --keep-empty                # Include empty hypotheses in scoring

Example:
    python score_st.py \
        --json /path/to/predictions.jsonl \
        --use-comet \
        --lower --rm-punctuation

Typical output:
    === TEXT PREDICTIONS (pred_text) ===
    Count: 4987 / 5000
    BLEU:   28.54
    chrF++: 52.76
    COMET:  0.631

    === ASR PREDICTIONS (speech_pred_transcribed) ===
    Count: 4950 / 5000
    BLEU:   26.91
    chrF++: 50.20
    COMET:  0.607

------------------------------------------------------------
NOTES
------------------------------------------------------------
• COMET requires GPU support and `pip install unbabel-comet`.
• BLEU/chrF++ use SacreBLEU's official tokenization and signature.
• For concatenated JSON (“}{”) format, the parser automatically splits objects.
• Text cleaning can be enabled for English-only normalization.

Author: Amir Hussein
Last modified: 2025-11-13
"""


import argparse
import json
from json.decoder import JSONDecoder
from typing import List, Dict, Tuple
from sacrebleu.metrics import BLEU, CHRF
import torch
import string
import re

try:
    from comet import load_from_checkpoint
    _HAS_COMET = True
except Exception:
    _HAS_COMET = False

def clean_english_text_keep_punct(s: str) -> str:
    if s is None:
        return ""

    # Strip leading/trailing whitespace first
    s = s.strip()

    # Remove non-ASCII characters (e.g., Chinese, emojis, etc.)
    s = re.sub(r"[^\x20-\x7E]", " ", s)

    # Collapse multiple spaces into one
    s = re.sub(r"\s+", " ", s)

    return s.strip()

def read_concat_json(file_path: str) -> List[Dict]:
    """Reads a file containing concatenated JSON objects (}{) or JSONL."""
    records = []
    dec = JSONDecoder()
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read().strip()

    i = 0
    n = len(data)
    while i < n:
        while i < n and data[i].isspace():
            i += 1
        if i >= n:
            break
        obj, idx = dec.raw_decode(data, i)
        records.append(obj)
        i = idx
    return records


def extract_refs_hyps(records: List[Dict], hyp_field: str, skip_empty=True, lower=False, rm_punctuation=False, clean_english_text=False):
    """Build lists of references and hypotheses from a specific hypothesis field."""
    refs, hyps, kept = [], [], []
    for idx, r in enumerate(records):
        ref = r.get("target_text", "")
        hyp = r.get(hyp_field, "")
        if lower:
            ref = ref.lower()
            hyp = hyp.lower()
        if rm_punctuation:
            ref = ref.translate(str.maketrans('', '', string.punctuation))
            hyp = hyp.translate(str.maketrans('', '', string.punctuation))
        if clean_english_text:
            ref = clean_english_text_keep_punct(ref)
            hyp = clean_english_text_keep_punct(hyp)
        if skip_empty and (hyp is None or str(hyp).strip() == ""):
            continue
        refs.append(str(ref).strip())
        hyps.append(str(hyp).strip())
        kept.append(idx)
    return refs, hyps, kept


def compute_bleu_chrf(hypotheses: List[str], references: List[str]):
    """Compute BLEU & chrF++ using SacreBLEU."""
    bleu = BLEU()
    chrf = CHRF(beta=2)
    refs_wrapped = [references]
    bleu_res = bleu.corpus_score(hypotheses, refs_wrapped)
    chrf_res = chrf.corpus_score(hypotheses, refs_wrapped)
    return bleu_res, chrf_res


def compute_comet_scores(references, hypotheses, sources=None, model_ckpt=None, batch_size=128):
    """Compute COMET score."""
    if not _HAS_COMET:
        raise RuntimeError("COMET not installed. Please `pip install unbabel-comet`.")

    if sources is None:
        sources = [""] * len(hypotheses)
    model = load_from_checkpoint(model_ckpt) if model_ckpt else load_from_checkpoint()
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    gpus = 1 if torch.cuda.is_available() else 0
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return float(out.system_score)


def evaluate_set(records, hyp_field, args, lower=False, rm_punctuation=False, clean_english_text=False):
    """Helper: evaluates a single hypothesis field."""
    refs, hyps, kept = extract_refs_hyps(records, hyp_field, skip_empty=not args.keep_empty, lower=lower, rm_punctuation=rm_punctuation, clean_english_text=clean_english_text)
    if len(hyps) == 0:
        return None

    bleu, chrf = compute_bleu_chrf(hyps, refs)
    comet = None
    if args.use_comet:
        comet = compute_comet_scores(refs, hyps, model_ckpt=args.comet_ckpt)
    return {
            "kept": len(hyps),
            "total": len(records),
            "bleu": bleu,
            "chrf": chrf,
            "comet": comet,
        }


def print_results(name, result):
    """Nicely format the results."""
    if result is None:
        print(f"\n=== {name.upper()} ===")
        print("No non-empty hypotheses found.")
        return
    print(f"\n=== {name.upper()} ===")
    print(f"Count: {result['kept']} / {result['total']}")
    print(f"BLEU:   {result['bleu']}")
    print(f"chrF++: {result['chrf']}")
    if result["comet"] is not None:
        print(f"COMET:  {result['comet']:.3f}")



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json", type=str, required=True, help="Path to concatenated JSON file.")
    parser.add_argument("--use-comet", action="store_true", default=True, help="Also compute COMET scores.")
    parser.add_argument("--comet-ckpt", type=str, default="pretrained_models/comet/model.ckpt", help="Path to COMET checkpoint.")
    parser.add_argument("--lower", dest="lower", action="store_true", default=True,
                    help="Lowercase ref/hyp before scoring (default: enabled).")
    parser.add_argument("--keep-empty", action="store_true", default=False, help="Include empty hypotheses in scoring.")
    parser.add_argument("--rm-punctuation", action="store_true", default=True, help="Remove punctuation from ref/hyp before scoring.")
    parser.add_argument("--clean-english-text", action="store_true", default=True, help="Clean English text and keep punctuation.")
    args = parser.parse_args()

    records = read_concat_json(args.json)
    if not records:
        raise ValueError("No valid JSON records found in the file!")

    # Evaluate both sets
    res_text = evaluate_set(records, "pred_text", args, lower=args.lower, rm_punctuation=args.rm_punctuation, clean_english_text=args.clean_english_text)
    res_asr = evaluate_set(records, "speech_pred_transcribed", args, lower=True, rm_punctuation=True, clean_english_text=args.clean_english_text)

    # Print results
    print_results("Text Predictions (pred_text)", res_text)
    print_results("ASR Predictions (speech_pred_transcribed)", res_asr)


if __name__ == "__main__":
    main()
