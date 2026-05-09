#!/usr/bin/env python3
# txtgrid2ctm_es.py — Praatio 6.x → CTM with "NA lex NA" tail

from praatio import textgrid
import os, argparse, sys
import string
import re

def read_txt(txt_file_path):
    words = []
    with open(txt_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
    return words

def remove_punc(txt):
    punct = string.punctuation + "¡¿"
    txt = re.sub(f"[{re.escape(punct)}]", "", txt)
    return txt

def rm_dash_at_end(txt):
    # remove dashes at the end of a word
    return txt.rstrip("-_")

def classify_token(token: str) -> str:
    if (token.startswith("'") or token.endswith("'")) and len(token.replace("'","")) < 3 :
        return "fragment"      # e.g., "'ve"
    elif re.search(r"\w+'\w+", token):
        return "contraction"   # e.g., "they've"
    else:
        return "other"

def find_tier(tg, name):
    for tier in tg.tiers:
        if tier.name == name:
            return tier
    raise ValueError(f"Tier '{name}' not found. Available: {[t.name for t in tg.tiers]}")

def textgrid_to_ctm(tg_path: str,
                    tier_name: str = "words",
                    channel: int = 1,
                    out_path: str =None,
                    keep_empty: bool = False,
                    decimals: int = 2):
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
    tier = find_tier(tg, tier_name)

    utt_id = os.path.splitext(os.path.basename(tg_path))[0]
    tg_dir = os.path.dirname(tg_path)
    ctm_file = os.path.join(tg_dir, f"{utt_id}.{tier_name}.ctm")
    if out_path is None:
        out_path = ctm_file

    fmt = f"{{:.{decimals}f}}"
    txt_path = os.path.join(tg_dir, f"{utt_id}.txt")
    origin_txt = read_txt(txt_path)
    if not origin_txt:
        print(f"Skipping {tg_path}: corresponding TXT file is empty.")
        return
    idx = 0
    oov = 0
    ctm_txt = []
    with open(out_path, "w", encoding="utf-8") as f:
        for start, end, label in tier.entries:
            label = (label or "").strip()
            
            if not label and not keep_empty or label == '<eps>' or classify_token(label) == "fragment":
                continue
            start_f = float(start)
            dur_f = max(0.0, float(end) - start_f)
            
            # Exact format: utt chan start dur token NA lex NA
            if idx < len(origin_txt) and (remove_punc(label) in remove_punc(origin_txt[idx].lower()) or label == '<unk>'):
                # origin_txt[idx] = rm_dash_at_end(origin_txt[idx])
                f.write(
                    f"{utt_id} {channel} {fmt.format(start_f)} {fmt.format(dur_f)} {origin_txt[idx]} NA lex NA\n"
                )
                if label == '<unk>':
                    oov += 1
                ctm_txt.append(label)
                idx += 1
            else:
                if idx < len(origin_txt):
                    print(f"id: {tg_path}")
                    print(f"label: {label} is different from origin {remove_punc(origin_txt[idx].lower())}")
            
    print(f"origin_txt: {origin_txt}")
    print(f"ctm: {ctm_txt}")
    print(f"OOV = {oov}, out of total:{len(origin_txt)}")

def main():
    p = argparse.ArgumentParser(description="Convert TextGrid to CTM with 'NA lex NA' tail.")
    p.add_argument("--txt-grid", required=True, help="Path to the TextGrid file")
    p.add_argument("--tier", default="words", help="Tier name (e.g., 'words' or 'phones')")
    p.add_argument("--channel", type=int, default=1, help="CTM channel (mono often 1; some tools expect 0)")
    p.add_argument("--out", default=None, help="Output CTM path (defaults to <utt>.<tier>.ctm)")
    p.add_argument("--keep-empty", action="store_true", help="Emit empty intervals as blank tokens")
    p.add_argument("--decimals", type=int, default=2, help="Decimal places for start/duration (default 2)")
    args = p.parse_args()

    textgrid_to_ctm(
        tg_path=args.txt_grid,
        tier_name=args.tier,
        channel=args.channel,
        out_path=args.out,
        keep_empty=args.keep_empty,
        decimals=args.decimals,
    )
    # print(f"Saved: {out_p}")

if __name__ == "__main__":
    main()
