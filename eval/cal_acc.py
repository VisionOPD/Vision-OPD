import argparse
import json
import sys
from collections import defaultdict


def is_correct(item):
    return str(item.get("judge", "")).strip().lower() == "yes"


def acc_text(correct, total):
    pct = (100.0 * correct / total) if total else 0.0
    return f"{correct}/{total} = {pct:.2f}%"


def calc_vstar(judge_json, benchmark):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in data:
        category = str(item.get("category", "unknown") or "unknown")
        category_stats[category]["total"] += 1
        if is_correct(item):
            overall_correct += 1
            category_stats[category]["correct"] += 1

    for category, stats in category_stats.items():
        total = stats["total"]
        acc = (100.0 * stats["correct"] / total) if total else 0.0
        print(f"{category}: {acc:.2f}% (n={total})")

    overall_acc = (100.0 * overall_correct / len(data)) if data else 0.0
    print(f"{benchmark} Acc: {overall_acc:.2f}% (n={len(data)})")


def calc_hrbench(judge_json, benchmark, benchmark_json):
    with open(judge_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(benchmark_json, "r", encoding="utf-8") as f:
        benchmark_records = json.load(f)

    def image_key(item):
        images = item.get("images") or []
        return str(images[0]) if isinstance(images, list) and images else ""

    def text_key(item):
        return (str(item.get("query", "") or ""), str(item.get("response", "") or ""))

    category_by_image = {}
    category_by_text = {}
    for item in benchmark_records:
        category = str(item.get("category", "unknown") or "unknown")
        img = image_key(item)
        txt = text_key(item)
        if img:
            category_by_image[img] = category
        category_by_text[txt] = category

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in records:
        correct = is_correct(item)
        category = str(item.get("category", "") or "").strip()
        if not category:
            category = category_by_image.get(image_key(item)) or category_by_text.get(text_key(item)) or "unknown"
        category_stats[category]["total"] += 1
        if correct:
            overall_correct += 1
            category_stats[category]["correct"] += 1

    print(f"{benchmark} Acc: {acc_text(overall_correct, len(records))}")


def calc_mme_realworld(judge_json, benchmark, benchmark_json):
    with open(judge_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(benchmark_json, "r", encoding="utf-8") as f:
        benchmark_records = json.load(f)

    def image_key(item):
        images = item.get("images") or []
        return str(images[0]) if isinstance(images, list) and images else ""

    def text_key(item):
        return (str(item.get("query", "") or ""), str(item.get("response", "") or ""))

    category_by_image = {}
    category_by_text = {}
    for item in benchmark_records:
        category = str(item.get("category", "unknown") or "unknown")
        img = image_key(item)
        txt = text_key(item)
        if img:
            category_by_image[img] = category
        category_by_text[txt] = category

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    supercategory_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0

    for item in records:
        correct = is_correct(item)
        category = str(item.get("category", "") or "").strip()
        if not category:
            category = category_by_image.get(image_key(item)) or category_by_text.get(text_key(item)) or "unknown"
        supercategory = category.split("/", 1)[0].strip() if "/" in category else category.strip()
        if not supercategory:
            supercategory = "unknown"
        category_stats[category]["total"] += 1
        supercategory_stats[supercategory]["total"] += 1
        if correct:
            overall_correct += 1
            category_stats[category]["correct"] += 1
            supercategory_stats[supercategory]["correct"] += 1

    print(f"{benchmark} Acc: {acc_text(overall_correct, len(records))}")


def calc_generic(judge_json, benchmark):
    with open(judge_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    c = sum(1 for x in data if is_correct(x))
    print(f"{benchmark} Acc: {c}/{n} = {100 * c / n:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy from judge results")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--judge_json", required=True, type=str, help="Path to judge output JSON")
    parser.add_argument("--benchmark_json", default=None, type=str, help="Path to original benchmark JSON (for category breakdown)")
    args = parser.parse_args()

    if args.benchmark == "vstar":
        calc_vstar(args.judge_json, args.benchmark)
    elif args.benchmark in ("hrbench-4k", "hrbench-8k"):
        if not args.benchmark_json:
            print("WARNING: --benchmark_json not provided, falling back to generic accuracy.", file=sys.stderr)
            calc_generic(args.judge_json, args.benchmark)
        else:
            calc_hrbench(args.judge_json, args.benchmark, args.benchmark_json)
    elif args.benchmark in ("mme-realworld", "mme-realworld-cn"):
        if not args.benchmark_json:
            print("WARNING: --benchmark_json not provided, falling back to generic accuracy.", file=sys.stderr)
            calc_generic(args.judge_json, args.benchmark)
        else:
            calc_mme_realworld(args.judge_json, args.benchmark, args.benchmark_json)
    else:
        calc_generic(args.judge_json, args.benchmark)


if __name__ == "__main__":
    main()
