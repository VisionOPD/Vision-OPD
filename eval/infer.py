import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

VSTAR_BENCHMARKS = {"vstar"}
VSTAR_MAX_IMAGE_SIZE = 20 * 1024 * 1024


def pick_image_path(item):
    return (item.get("images") or [""])[0]


def make_sample_uid(item, benchmark):
    for key in ("sample_uid", "uid", "question_id", "index", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{key}:{value}"
    stable_obj = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(stable_obj, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def should_retry_existing_record(item):
    model_answer = item.get("model_answer")
    if not isinstance(model_answer, str):
        return True
    model_answer = model_answer.strip()
    if not model_answer:
        return True
    return model_answer.startswith("[API_ERROR]") or model_answer.startswith("[FUTURE_ERROR]")


def compact_existing_output(path, benchmark):
    if not path.exists():
        return [], {}, False
    ordered_uids = []
    best_records = {}
    changed = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                changed = True
                continue
            try:
                record = json.loads(line)
            except Exception:
                changed = True
                continue
            sample_uid = record.get("sample_uid") or make_sample_uid(record, benchmark)
            record["sample_uid"] = sample_uid
            if sample_uid not in best_records:
                ordered_uids.append(sample_uid)
                best_records[sample_uid] = record
                continue
            changed = True
            prev = best_records[sample_uid]
            prev_is_error = should_retry_existing_record(prev)
            curr_is_error = should_retry_existing_record(record)
            if prev_is_error and not curr_is_error:
                best_records[sample_uid] = record
            elif prev_is_error == curr_is_error:
                best_records[sample_uid] = record
    return ordered_uids, best_records, changed


def rewrite_output(path, ordered_uids, record_map):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for sample_uid in ordered_uids:
            record = record_map.get(sample_uid)
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def normalize_model_answer(model_answer_raw):
    if not isinstance(model_answer_raw, str):
        return model_answer_raw
    start = model_answer_raw.rfind("<answer>")
    end = model_answer_raw.find("</answer>", start + len("<answer>")) if start != -1 else -1
    if start != -1 and end != -1 and end > start:
        return model_answer_raw[start + len("<answer>"):end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:"):].strip()
    return model_answer_raw.strip()


def image_to_data_uri(path_str, benchmark):
    if benchmark in VSTAR_BENCHMARKS:
        img = Image.open(path_str).convert("RGB")
        output = BytesIO()
        img.save(output, format="PNG")
        byte_data = output.getvalue()
        while len(byte_data) > VSTAR_MAX_IMAGE_SIZE and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            output = BytesIO()
            img.save(output, format="PNG")
            byte_data = output.getvalue()
        b64 = base64.b64encode(byte_data).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    p = Path(path_str)
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return f"data:{mime};base64,{b64}"


def main():
    parser = argparse.ArgumentParser(description="Run inference via OpenAI-compatible API")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--benchmark_json", required=True, type=str, help="Path to benchmark JSON file")
    parser.add_argument("--out_dir", default="model_answer", type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--api_base", required=True, type=str)
    parser.add_argument("--api_key", default="EMPTY", type=str)
    parser.add_argument("--model_id", required=True, type=str, help="OpenAI model ID")
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--parallel_workers", default=32, type=int)
    args = parser.parse_args()

    benchmark = args.benchmark
    data_path = Path(args.benchmark_json)
    with open(data_path, "r", encoding="utf-8") as f:
        total_data = json.load(f)

    out_dir = Path(args.out_dir) / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name}_answer.jsonl"

    done_ids = set()
    retry_ids = set()
    if out_path.exists():
        ordered_uids, existing_records, was_compacted = compact_existing_output(out_path, benchmark)
        if was_compacted:
            rewrite_output(out_path, ordered_uids, existing_records)
            print(f"Compacted checkpoint file to {len(existing_records)} unique samples.")
        for sample_uid in ordered_uids:
            x = existing_records[sample_uid]
            if should_retry_existing_record(x):
                retry_ids.add(sample_uid)
                continue
            done_ids.add(sample_uid)
        if retry_ids:
            print(f"Retrying {len(retry_ids)} samples with failed/empty existing records.")
        print(f"Checkpoint detected, {len(done_ids)} samples already completed.")

    todo_data = []
    for item in total_data:
        sample_uid = make_sample_uid(item, benchmark)
        if sample_uid in done_ids:
            continue
        record = dict(item)
        record["sample_uid"] = sample_uid
        todo_data.append(record)

    print(f"Remaining samples to process: {len(todo_data)}")
    if not todo_data:
        print("All samples have been processed.")
        return

    thread_local = threading.local()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=3600)
            thread_local.client = c
        return c

    def run_one(item):
        sample_uid = item.get("sample_uid") or make_sample_uid(item, benchmark)
        img_path = pick_image_path(item)
        query = item.get("query", "").replace("<image>", "").strip()
        data_uri = image_to_data_uri(img_path, benchmark)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": query},
                ],
            },
        ]
        model_answer = ""
        raw_model_answer = ""
        client = get_client()
        for attempt in range(1, args.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=0,
                )
                raw_model_answer = (resp.choices[0].message.content or "").strip()
                model_answer = normalize_model_answer(raw_model_answer)
                break
            except Exception as e:
                if attempt == args.max_retries:
                    raw_model_answer = f"[API_ERROR] {e}"
                    model_answer = raw_model_answer
                else:
                    time.sleep(1.0)
        record = dict(item)
        record["sample_uid"] = sample_uid
        record["model_answer"] = model_answer
        if raw_model_answer and raw_model_answer != model_answer:
            record["model_answer_raw"] = raw_model_answer
        return record

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, open(out_path, "a", encoding="utf-8") as f_out:
        future_to_item = {executor.submit(run_one, item): item for item in todo_data}
        with tqdm(total=len(todo_data), desc="Inference", unit="case", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_item):
                try:
                    record = future.result()
                except Exception as e:
                    item = future_to_item[future]
                    record = dict(item)
                    record["sample_uid"] = item.get("sample_uid") or make_sample_uid(item, benchmark)
                    record["model_answer"] = f"[FUTURE_ERROR] {e}"
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                pbar.update(1)

    elapsed = time.time() - start
    ordered_uids, final_records, was_compacted = compact_existing_output(out_path, benchmark)
    if was_compacted:
        rewrite_output(out_path, ordered_uids, final_records)
    print(f"Compacted final output to {len(final_records)} unique samples.")
    print(f"Inference done in {elapsed:.1f}s")
    print(f"Saved answers to: {out_path}")


if __name__ == "__main__":
    main()
