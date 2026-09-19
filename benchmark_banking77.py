"""Banking77 Benchmark for OpenSourceJev.

Designed for local execution on a 4GB VRAM GPU. Evaluates queries strictly
sequentially (batch size = 1) with automatic memory recycling.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app.models import RunRequest, Step
from app.native_engine import NativeJevEngine

# URLs for official Banking77 dataset from PolyAI
CATEGORIES_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json"
TEST_DATA_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
LOCAL_DATA_DIR = PROJECT_ROOT / "benchmark_data"
CATEGORIES_FILE = LOCAL_DATA_DIR / "categories.json"
TEST_FILE = LOCAL_DATA_DIR / "test.csv"


def download_banking77():
    LOCAL_DATA_DIR.mkdir(exist_ok=True)
    if not CATEGORIES_FILE.is_file():
        print(f"Downloading Banking77 categories...")
        urllib.request.urlretrieve(CATEGORIES_URL, str(CATEGORIES_FILE))
    if not TEST_FILE.is_file():
        print(f"Downloading Banking77 test.csv...")
        urllib.request.urlretrieve(TEST_DATA_URL, str(TEST_FILE))


def load_dataset() -> Tuple[List[str], List[Tuple[str, str]]]:
    download_banking77()
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        categories = json.load(f)

    samples = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                samples.append((row[0], row[1]))
    return categories, samples


def run_benchmark(
    num_samples: int = 20,
    num_classes: int = 10,
    model_path: str = "models/Qwen3-1.7B-Q8_0.gguf"
):
    all_categories, all_samples = load_dataset()

    if num_classes < len(all_categories):
        # Select first num_classes diverse categories
        categories = all_categories[:num_classes]
        eval_candidates = [s for s in all_samples if s[1] in categories]
    else:
        categories = all_categories
        eval_candidates = all_samples

    print(f"\n==================================================")
    print(f"    Banking77 Benchmark: OpenSourceJev")
    print(f"==================================================")
    print(f"Target Classes        : {len(categories)} (out of 77)")
    print(f"Candidate Test Pool   : {len(eval_candidates)} queries")
    print(f"Evaluating Samples    : {num_samples} (strictly 1-by-1, 4GB VRAM safe)")
    print(f"Model Path            : {model_path}")
    print(f"==================================================\n")

    # Select evenly spaced samples across the pool
    step_size = max(1, len(eval_candidates) // num_samples)
    eval_samples = [eval_candidates[i * step_size] for i in range(num_samples)]

    engine = NativeJevEngine()
    
    top1_correct = 0
    top3_correct = 0
    latencies = []
    confidences_correct = []
    confidences_incorrect = []

    criteria_map = {
        cat: cat.replace("_", " ") for cat in categories
    }

    start_total = time.perf_counter()

    for idx, (query, true_label) in enumerate(eval_samples, 1):
        step_t0 = time.perf_counter()
        
        request = RunRequest(
            context=f"Customer Query: \"{query}\"",
            mode="native",
            model_path=model_path,
            workflow=[
                Step(
                    id="intent",
                    kind="choice",
                    prompt="Classify the banking intent of this customer query into exactly one category:",
                    options=categories,
                    criteria=criteria_map,
                )
            ]
        )

        try:
            response = engine._run_sync(request)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            continue

        step_elapsed = (time.perf_counter() - step_t0) * 1000
        latencies.append(step_elapsed)

        trace = response.trace[0]
        candidates = trace.candidates
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
        top1 = ranked[0]["value"] if ranked else None
        top3 = [c["value"] for c in ranked[:3]]
        confidence = ranked[0]["score"] if ranked else 0.0

        is_top1 = (top1 == true_label)
        is_top3 = (true_label in top3)

        if is_top1:
            top1_correct += 1
            confidences_correct.append(confidence)
        else:
            confidences_incorrect.append(confidence)

        if is_top3:
            top3_correct += 1

        acc1 = (top1_correct / idx) * 100
        acc3 = (top3_correct / idx) * 100
        avg_lat = sum(latencies) / len(latencies)
        print(
            f"[{idx:2d}/{num_samples}] "
            f"Top-1: {acc1:5.1f}% | "
            f"Top-3: {acc3:5.1f}% | "
            f"Lat: {step_elapsed:6.1f}ms | "
            f"{'[OK]' if is_top1 else '[NO]'} "
            f"\"{query[:35]}...\" -> {top1} (True: {true_label})"
        )

    total_time = time.perf_counter() - start_total
    
    print("\n" + "=" * 50)
    print("           BENCHMARK RESULTS REPORT")
    print("=" * 50)
    print(f"Evaluated Samples     : {num_samples}")
    print(f"Category Count        : {len(categories)} categories")
    print(f"Total Benchmark Time  : {total_time:.2f}s")
    print(f"Average Latency       : {sum(latencies)/len(latencies):.1f} ms / query")
    print(f"Median Latency        : {sorted(latencies)[len(latencies)//2]:.1f} ms / query")
    print(f"Top-1 Accuracy        : {(top1_correct / num_samples)*100:.2f}% ({top1_correct}/{num_samples})")
    print(f"Top-3 Accuracy        : {(top3_correct / num_samples)*100:.2f}% ({top3_correct}/{num_samples})")
    if confidences_correct:
        print(f"Avg Confidence (Correct)   : {sum(confidences_correct)/len(confidences_correct):.4f}")
    if confidences_incorrect:
        print(f"Avg Confidence (Incorrect) : {sum(confidences_incorrect)/len(confidences_incorrect):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banking77 Benchmark for OpenSourceJev")
    parser.add_argument("--samples", type=int, default=20, help="Number of test queries to evaluate")
    parser.add_argument("--classes", type=int, default=10, help="Number of categories to evaluate (e.g. 10 or 77)")
    args = parser.parse_args()

    run_benchmark(num_samples=args.samples, num_classes=args.classes)
