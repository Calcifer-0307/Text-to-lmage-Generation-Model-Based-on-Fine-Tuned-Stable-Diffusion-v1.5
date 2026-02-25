import argparse
import os
import csv
from typing import Dict, Any, List
from datasets import load_dataset
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Merge corrected CSV labels into HF dataset loaded from local cache and save to disk.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the corrected CSV file.")
    parser.add_argument("--cache-dir", type=str, required=True, help="HF datasets cache dir containing moonworks/lunara-aesthetic.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the merged dataset (save_to_disk).")
    parser.add_argument("--revision", type=str, default="851305085843a2b2d96ea0d44904bc54a670c5f4", help="Hub revision hash to pin the original dataset snapshot.")
    parser.add_argument("--offline", action="store_true", help="Load strictly from local cache without contacting Hub.")
    return parser.parse_args()

def load_corrections(csv_path: str) -> Dict[int, Dict[str, Any]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    correction_map: Dict[int, Dict[str, Any]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        allowed = {"prompt", "region", "category", "topic"}
        for row in reader:
            if "index" not in row or row["index"] is None or row["index"] == "":
                continue
            try:
                idx = int(row["index"])
            except ValueError:
                continue
            updates = {k: v for k, v in row.items() if k in allowed and v is not None}
            correction_map[idx] = updates
    return correction_map

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading corrections from {args.csv_path} ...")
    corrections = load_corrections(args.csv_path)
    print(f"Loaded {len(corrections)} correction entries.")

    print(f"Loading dataset from cache dir {args.cache_dir} (revision {args.revision}) ...")
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        parquet_files = sorted(glob.glob(os.path.join(args.cache_dir, "**", "train-*-of-*.parquet"), recursive=True))
        arrow_files = sorted(glob.glob(os.path.join(args.cache_dir, "**", "*.arrow"), recursive=True))
        if parquet_files:
            ds = load_dataset("parquet", data_files=parquet_files, split="train")
        elif arrow_files:
            ds = load_dataset("arrow", data_files=arrow_files, split="train")
        else:
            raise FileNotFoundError("No local shards (.parquet or .arrow) found in cache-dir while offline.")
    else:
        ds = load_dataset("moonworks/lunara-aesthetic", split="train", cache_dir=args.cache_dir, revision=args.revision, download_mode="reuse_cache_if_exists")
    print(f"Dataset loaded: {len(ds)} rows.")

    print("Applying corrections ...")
    def apply_updates(example, idx):
        upd = corrections.get(idx)
        if upd:
            for k, v in upd.items():
                if v is not None and v != "":
                    example[k] = v
        return example

    merged = ds.map(apply_updates, with_indices=True)

    changed_count = 0
    sample_indices: List[int] = []
    for i in range(min(100, len(merged))):
        if i in corrections:
            changed_count += 1
            sample_indices.append(i)
    print(f"Sanity check: {changed_count} of first {min(100, len(merged))} rows had corrections.")
    if sample_indices:
        print(f"Sample corrected indices (first 10): {sample_indices[:10]}")

    print(f"Saving merged dataset to {args.output_dir} ...")
    merged.save_to_disk(args.output_dir)
    print("Done. You can load it via datasets.load_from_disk(output_dir).")

if __name__ == "__main__":
    main()
