import json
import argparse
import ndjson
import os
import shutil

def write_ndjson(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        ndjson.dump(data, f, ensure_ascii=True)

def fix_ascii_jsonl(manifest1: str, manifest2: str, field1: str = "text_translated", field2: str = "text"):
    
    
    count = 0
    data = []
    dict1 = {}
    dict2 = {}
    manifest_name = os.path.basename(manifest1)
    tmp_output = f"/tmp/fixed_{manifest_name}"
    with open(manifest1, "r", encoding="utf-8") as f1:
        for line1 in f1:
            sample1 = json.loads(line1)
            name_without_ext = os.path.splitext(os.path.basename(sample1["audio_filepath"]))[0]
            dict1[name_without_ext] = sample1
    with open(manifest2, "r", encoding="utf-8") as f2:
        for line2 in f2:
            sample2 = json.loads(line2)
            name_without_ext = os.path.splitext(os.path.basename(sample2["audio_filepath"]))[0]
            dict2[name_without_ext] = sample2
    for id2 in dict2.keys():
        if not dict2[id2][field2].isascii():
            dict2[id2][field2] = dict1[id2][field1]
        data.append(dict2[id2])
            # try:
            #     sample = json.loads(line)
            #     text = sample.get(field, "")
            #     if not text.isascii():
            #         count += 1
            #         print(f"[{sample['audio_filepath']}] Non-ASCII in {field}: {repr(text)}")
            # except json.JSONDecodeError as e:
            #     print(f"Error parsing line: {e}")
    if len(data) != len(dict2):
        raise ValueError(f"Length of data ({len(data)}) does not match length of dict2 ({len(dict2)})")
    else:
        # print(f"SUCCESS: {manifest_name}")
        write_ndjson(data, tmp_output)
        shutil.move(tmp_output, manifest2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for non-ASCII characters in manifest field")
    parser.add_argument("--manifest1", type=str, required=True, help="Path to the reference.jsonl manifest file")
    parser.add_argument("--manifest2", type=str, required=True, help="Path to the target.jsonl manifest file")
    parser.add_argument("--field1", type=str, default="text_translated", help="Field to check (default: 'text_translated')")
    parser.add_argument("--field2", type=str, default="text", help="Field to check (default: 'text')")

    args = parser.parse_args()

    fix_ascii_jsonl(args.manifest1, args.manifest2, field1=args.field1, field2=args.field2)
