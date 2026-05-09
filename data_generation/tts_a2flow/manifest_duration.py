from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator
from lhotse import CutSet
import os
from pathlib import Path


def load_nemo_tarred_from_dir(manifest_path: str, tar_paths: str) -> CutSet:
    """
    Load cuts from tarred files
    """
    # Initialize iterator
    iterator = LazyNeMoTarredIterator(
                        manifest_path=manifest_path,
                        tar_paths=tar_paths,
                        allow_skipme=True,
                        skip_missing_manifest_entries=True,
                    )
    return iterator

def load_all_tarred_manifests(manifest_dir: str, tar_dir: str):
    manifest_files = sorted(Path(manifest_dir).rglob("*.jsonl"))
    tar_files = sorted(Path(tar_dir).rglob("*.tar"))

    if not manifest_files:
        raise FileNotFoundError(f"No manifest JSON files found in {manifest_dir}")
    if not tar_files:
        raise FileNotFoundError(f"No tar files found in {tar_dir}")

    print(f"Found {len(manifest_files)} manifests and {len(tar_files)} tar files.")

    # Assuming manifests and tars follow the same order/indexing
    manifests = [str(manifest_file) for manifest_file in manifest_files]
    tars = [str(tar_file) for tar_file in tar_files]
    iterator = load_nemo_tarred_from_dir(manifests, tars)
    cuts = CutSet(iterator)
    cuts = cuts.filter(lambda x: x.custom.get('reason') != 'LengthRatioFilter')
    durations = [float(c.duration) for c in cuts]
    filtered_cuts = cuts.filter(lambda x: 1<= x.duration <= 20)
    filtered_durations = [float(c.duration) for c in filtered_cuts]
    return durations, filtered_durations


if __name__ == "__main__":
    data_root = "/lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/AST/data/train/v2/riva_asr_es-US_v1.0"
    bucket = 1
    tar_paths = f"{data_root}/bucket{bucket}"
    manifest_paths = f"{data_root}/bucket{bucket}/sharded_manifests_es-US_en-US"
    all_durations, filtered_durations = load_all_tarred_manifests(manifest_paths, tar_paths)
    print(f"Total duration: {sum(all_durations)/3600:.2f} hours")
    print(f"Total duration after filtering: {sum(filtered_durations)/3600:.2f} hours")
    print(f"Total filtered duration: {(sum(all_durations) - sum(filtered_durations))/3600:.2f} hours")

    