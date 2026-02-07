import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from data_check import load, check_schema, analyze, save_report

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_bar(dist: Dict[str, int], title: str, out_path: str) -> None:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def table_md(dist: Dict[str, int], header: Tuple[str, str]) -> str:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    lines: List[str] = []
    lines.append(f"| {header[0]} | {header[1]} |")
    lines.append(f"|---|---:|")
    for k, v in items:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

def render_markdown(data: Dict[str, Any], fig_dir: str) -> Tuple[str, List[str]]:
    ensure_dir(fig_dir)

    c_cat_png = os.path.join(fig_dir, "corrected_category_distribution.png")
    c_reg_png = os.path.join(fig_dir, "corrected_region_distribution.png")
    c_top_png = os.path.join(fig_dir, "corrected_topic_distribution.png")
    plot_bar(data.get("corrected_category_distribution", {}), "Corrected Category Distribution", c_cat_png)
    plot_bar(data.get("corrected_region_distribution", {}), "Corrected Region Distribution", c_reg_png)
    plot_bar(data.get("corrected_topic_distribution", {}), "Corrected Topic Distribution", c_top_png)
    lines: List[str] = []
    lines.append(f"# Data Check Report")
    lines.append("")
    lines.append(f"- Checked rows: {data.get('checked_rows')}")
    lines.append(f"- Schema OK: {data.get('schema_ok')}")
    lines.append("")
    pl = data.get("prompt_length", {})
    lines.append("## Prompt Length Stats")
    lines.append(f"- Min: {pl.get('min')}")
    lines.append(f"- Max: {pl.get('max')}")
    lines.append(f"- Avg: {pl.get('avg')}")
    lines.append("")
    im = data.get("image_size", {})
    lines.append("## Image Size Stats")
    lines.append(f"- Width: min={im.get('min_width')} max={im.get('max_width')} avg={im.get('avg_width')}")
    lines.append(f"- Height: min={im.get('min_height')} max={im.get('max_height')} avg={im.get('avg_height')}")
    lines.append("")

    lines.append("## Corrected Category Distribution")
    lines.append(f"![Corrected Category Distribution]({c_cat_png})")
    lines.append("")
    lines.append(table_md(data.get("corrected_category_distribution", {}), ("Category", "Count")))
    lines.append("")
    lines.append("## Corrected Region Distribution")
    lines.append(f"![Corrected Region Distribution]({c_reg_png})")
    lines.append("")
    lines.append(table_md(data.get("corrected_region_distribution", {}), ("Region", "Count")))
    lines.append("")
    lines.append("## Corrected Topic Distribution")
    lines.append(f"![Corrected Topic Distribution]({c_top_png})")
    lines.append("")
    lines.append(table_md(data.get("corrected_topic_distribution", {}), ("Topic", "Count")))
    lines.append("")
    lines.append("## Applied Corrections")
    corr_stats = data.get("correction_stats", [])
    if not corr_stats:
        lines.append("_No corrections applied_")
    else:
        lines.append("| Field | From | To | Count |")
        lines.append("|---|---|---|---:|")
        for it in corr_stats:
            lines.append(f"| {it.get('field')} | {it.get('from')} | {it.get('to')} | {it.get('count')} |")
    lines.append("")
    md_content = "\n".join(lines)
    return md_content, [c_cat_png, c_reg_png, c_top_png]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/lunara_check_report.json", type=str)
    parser.add_argument("--out", default="data/lunara_check_report.md", type=str)
    parser.add_argument("--fig-dir", default="data/report_figs", type=str)
    args = parser.parse_args()
    if not os.path.exists(args.input):
        ds = load("train[:100]", "./data")
        schema_ok, missing = check_schema(ds.column_names)
        report = analyze(ds, 100)
        report["schema_ok"] = schema_ok
        report["schema_missing"] = missing
        save_report(args.input, report)
    data = load_json(args.input)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        md, _ = render_markdown(data, args.fig_dir)
        f.write(md)
    print(f"Generated Markdown report: {args.out}, figures: {args.fig_dir}")

if __name__ == "__main__":
    main()
