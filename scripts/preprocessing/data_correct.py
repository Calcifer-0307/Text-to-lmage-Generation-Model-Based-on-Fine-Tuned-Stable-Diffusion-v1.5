import os
import csv
import json
import argparse
from typing import Dict, Any, List
from difflib import SequenceMatcher
from datasets import load_dataset

def load(split: str, cache_dir: str):
    return load_dataset("moonworks/lunara-aesthetic", split=split, cache_dir=cache_dir)

def build_distributions(ds) -> Dict[str, Dict[str, int]]:
    cat: Dict[str, int] = {}
    reg: Dict[str, int] = {}
    top: Dict[str, int] = {}
    for i in range(len(ds)):
        r = ds[i]
        c = r.get("category")
        rg = r.get("region")
        tp = r.get("topic")
        if isinstance(c, str):
            cat[c] = cat.get(c, 0) + 1
        if isinstance(rg, str):
            reg[rg] = reg.get(rg, 0) + 1
        if isinstance(tp, str):
            top[tp] = top.get(tp, 0) + 1
    return {"category": cat, "region": reg, "topic": top}

def build_suspects(dist: Dict[str, int]) -> List[Dict[str, Any]]:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    suspects: List[Dict[str, Any]] = []
    if not items:
        return suspects
    top_counts = {label: count for label, count in items[:max(5, len(items))]}
    for label, count in items:
        for cand, ccount in top_counts.items():
            if cand == label:
                continue
            ratio = SequenceMatcher(None, label.lower(), cand.lower()).ratio()
            if ratio >= 0.88 and ccount >= max(2, count * 2):
                suspects.append({"value": label, "count": count, "suggest": cand, "suggest_count": ccount, "similarity": round(ratio, 3)})
                break
    return suspects

def build_correction_map(dists: Dict[str, int]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {"category": {}, "region": {}, "topic": {}}
    for field in ["category", "region", "topic"]:
        sus = build_suspects(dists[field])
        for it in sus:
            mapping[field][it["value"]] = it["suggest"]
    return mapping

def correct_and_save(ds, mapping: Dict[str, Dict[str, str]], out_csv: str, summary_json: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    corr_stats: Dict[str, int] = {}
    c_cat: Dict[str, int] = {}
    c_reg: Dict[str, int] = {}
    c_top: Dict[str, int] = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "prompt", "region", "category", "topic"])
        for i in range(len(ds)):
            r = ds[i]
            region = r.get("region")
            category = r.get("category")
            topic = r.get("topic")
            prompt = r.get("prompt")
            new_region = mapping.get("region", {}).get(region, region) if isinstance(region, str) else region
            new_category = mapping.get("category", {}).get(category, category) if isinstance(category, str) else category
            new_topic = mapping.get("topic", {}).get(topic, topic) if isinstance(topic, str) else topic
            if isinstance(category, str) and new_category != category:
                key = f"category:{category}->{new_category}"
                corr_stats[key] = corr_stats.get(key, 0) + 1
            if isinstance(region, str) and new_region != region:
                key = f"region:{region}->{new_region}"
                corr_stats[key] = corr_stats.get(key, 0) + 1
            if isinstance(topic, str) and new_topic != topic:
                key = f"topic:{topic}->{new_topic}"
                corr_stats[key] = corr_stats.get(key, 0) + 1
            if isinstance(new_category, str):
                c_cat[new_category] = c_cat.get(new_category, 0) + 1
            if isinstance(new_region, str):
                c_reg[new_region] = c_reg.get(new_region, 0) + 1
            if isinstance(new_topic, str):
                c_top[new_topic] = c_top.get(new_topic, 0) + 1
            w.writerow([i, prompt, new_region, new_category, new_topic])
    corr_list: List[Dict[str, Any]] = []
    for k, v in corr_stats.items():
        field, pair = k.split(":", 1)
        src, dst = pair.split("->", 1)
        corr_list.append({"field": field, "from": src, "to": dst, "count": v})
    summary = {"correction_map": mapping, "corrected_category_distribution": c_cat, "corrected_region_distribution": c_reg, "corrected_topic_distribution": c_top, "correction_stats": corr_list}
    with open(summary_json, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--cache-dir", default="./data", type=str)
    parser.add_argument("--out-csv", default="data/lunara_corrected_labels.csv", type=str)
    parser.add_argument("--summary", default="data/lunara_correction_summary.json", type=str)
    args = parser.parse_args()
    ds = load(args.split, args.cache_dir)
    dists = build_distributions(ds)
    mapping = build_correction_map(dists)
    correct_and_save(ds, mapping, args.out_csv, args.summary)
    print(f"Saved corrected labels CSV: {args.out_csv}, summary: {args.summary}")

if __name__ == "__main__":
    main()
