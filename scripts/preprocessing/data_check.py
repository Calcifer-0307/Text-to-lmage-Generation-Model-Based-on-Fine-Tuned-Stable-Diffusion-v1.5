import os
import json
import argparse
from collections import Counter
from statistics import mean
from typing import Dict, Any, List, Tuple
from difflib import SequenceMatcher
from datasets import load_dataset
from PIL import Image

def load(split: str, cache_dir: str) -> Any:
    return load_dataset("moonworks/lunara-aesthetic", split=split, cache_dir=cache_dir)

def check_schema(column_names: List[str]) -> Tuple[bool, List[str]]:
    expected = ["image", "prompt", "region", "category", "topic"]
    missing = [c for c in expected if c not in column_names]
    return len(missing) == 0, missing

def analyze(ds: Any, sample_size: int) -> Dict[str, Any]:
    size = min(sample_size, len(ds))
    missing_counts = Counter()
    invalid_counts = Counter()
    prompt_lengths: List[int] = []
    widths: List[int] = []
    heights: List[int] = []
    category_counts = Counter()
    region_counts = Counter()
    topic_counts = Counter()
    issues: List[Dict[str, Any]] = []
    for i in range(size):
        row = ds[i]
        img = row.get("image")
        prompt = row.get("prompt")
        region = row.get("region")
        category = row.get("category")
        topic = row.get("topic")
        if img is None:
            missing_counts["image"] += 1
        elif not isinstance(img, Image.Image):
            invalid_counts["image"] += 1
        else:
            w, h = img.size
            widths.append(w)
            heights.append(h)
        if prompt is None or (isinstance(prompt, str) and prompt.strip() == ""):
            missing_counts["prompt"] += 1
        elif not isinstance(prompt, str):
            invalid_counts["prompt"] += 1
        else:
            prompt_lengths.append(len(prompt))
        if region is None or (isinstance(region, str) and region.strip() == ""):
            missing_counts["region"] += 1
        elif not isinstance(region, str):
            invalid_counts["region"] += 1
        else:
            region_counts[region] += 1
        if category is None or (isinstance(category, str) and category.strip() == ""):
            missing_counts["category"] += 1
        elif not isinstance(category, str):
            invalid_counts["category"] += 1
        else:
            category_counts[category] += 1
        if topic is None or (isinstance(topic, str) and topic.strip() == ""):
            missing_counts["topic"] += 1
        elif not isinstance(topic, str):
            invalid_counts["topic"] += 1
        else:
            topic_counts[topic] += 1
        if invalid_counts or missing_counts:
            pass
    report: Dict[str, Any] = {}
    report["checked_rows"] = size
    report["missing_counts"] = dict(missing_counts)
    report["invalid_counts"] = dict(invalid_counts)
    report["prompt_length"] = {
        "min": min(prompt_lengths) if prompt_lengths else None,
        "max": max(prompt_lengths) if prompt_lengths else None,
        "avg": round(mean(prompt_lengths), 2) if prompt_lengths else None,
    }
    report["image_size"] = {
        "min_width": min(widths) if widths else None,
        "max_width": max(widths) if widths else None,
        "avg_width": round(mean(widths), 2) if widths else None,
        "min_height": min(heights) if heights else None,
        "max_height": max(heights) if heights else None,
        "avg_height": round(mean(heights), 2) if heights else None,
    }
    report["category_distribution"] = dict(category_counts)
    report["region_distribution"] = dict(region_counts)
    report["topic_distribution"] = dict(topic_counts)
    report["issues"] = issues
    return report

def _spelling_suspects(dist: Dict[str, int]) -> List[Dict[str, Any]]:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    suspects: List[Dict[str, Any]] = []
    if not items:
        return suspects
    top_counts = {label: count for label, count in items[:max(5, len(items))]}
    for label, count in items:
        if count == 0:
            continue
        for cand, ccount in top_counts.items():
            if cand == label:
                continue
            ratio = SequenceMatcher(None, label.lower(), cand.lower()).ratio()
            if ratio >= 0.88 and ccount >= max(2, count * 2):
                suspects.append({
                    "value": label,
                    "count": count,
                    "suggest": cand,
                    "suggest_count": ccount,
                    "similarity": round(ratio, 3),
                })
                break
    return suspects

def _build_correction_map(suspects: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for it in suspects:
        src = it.get("value")
        dst = it.get("suggest")
        if isinstance(src, str) and isinstance(dst, str):
            mapping[src] = dst
    return mapping

def apply_corrections(ds: Any, size: int, correction_map: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], List[Dict[str, Any]]]:
    cat_counts = Counter()
    reg_counts = Counter()
    top_counts = Counter()
    corr_stats: Counter = Counter()
    for i in range(size):
        row = ds[i]
        category = row.get("category")
        region = row.get("region")
        topic = row.get("topic")
        if isinstance(category, str):
            new_cat = correction_map.get("category", {}).get(category, category)
            if new_cat != category:
                corr_stats[f"category:{category}->{new_cat}"] += 1
            cat_counts[new_cat] += 1
        if isinstance(region, str):
            new_reg = correction_map.get("region", {}).get(region, region)
            if new_reg != region:
                corr_stats[f"region:{region}->{new_reg}"] += 1
            reg_counts[new_reg] += 1
        if isinstance(topic, str):
            new_top = correction_map.get("topic", {}).get(topic, topic)
            if new_top != topic:
                corr_stats[f"topic:{topic}->{new_top}"] += 1
            top_counts[new_top] += 1
    corr_list: List[Dict[str, Any]] = []
    for key, cnt in corr_stats.items():
        field, pair = key.split(":", 1)
        src, dst = pair.split("->", 1)
        corr_list.append({"field": field, "from": src, "to": dst, "count": cnt})
    return dict(cat_counts), dict(reg_counts), dict(top_counts), corr_list

def save_report(path: str, content: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--sample-size", default=-1, type=int)
    parser.add_argument("--cache-dir", default="./data", type=str)
    parser.add_argument("--out", default="data/lunara_check_report.json", type=str)
    args = parser.parse_args()
    ds = load(args.split, args.cache_dir)
    schema_ok, missing = check_schema(ds.column_names)
    size = len(ds) if args.sample_size is None or args.sample_size < 0 else args.sample_size
    report = analyze(ds, size)
    report["schema_ok"] = schema_ok
    report["schema_missing"] = missing
    spelling = {
        "category": _spelling_suspects(report["category_distribution"]),
        "region": _spelling_suspects(report["region_distribution"]),
        "topic": _spelling_suspects(report["topic_distribution"]),
    }
    report["spelling_suspects"] = spelling
    correction_map = {
        "category": _build_correction_map(spelling["category"]),
        "region": _build_correction_map(spelling["region"]),
        "topic": _build_correction_map(spelling["topic"]),
    }
    report["correction_map"] = correction_map
    c_cat, c_reg, c_top, c_stats = apply_corrections(ds, size, correction_map)
    report["corrected_category_distribution"] = c_cat
    report["corrected_region_distribution"] = c_reg
    report["corrected_topic_distribution"] = c_top
    report["correction_stats"] = c_stats
    save_report(args.out, report)
    print(f"已检查 {report['checked_rows']} 条样本（全量={size==len(ds)}），schema_ok={report['schema_ok']}，报告已保存到 {args.out}")

if __name__ == "__main__":
    main()
