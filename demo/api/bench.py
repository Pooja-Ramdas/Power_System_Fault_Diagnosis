"""
demo/api/bench.py
Benchmark the live API with 50 sequential + 10 concurrent POST /diagnose calls.
Run after: uvicorn demo.api.main:app --host 0.0.0.0 --port 8000

Usage:
    python demo/api/bench.py [--url http://localhost:8000] [--case CASE-0001]
"""

import time
import json
import argparse
import urllib.request
import urllib.error
import threading
from pathlib import Path

DEFAULT_URL = "http://localhost:8000"
INDEX_FILE = Path(__file__).resolve().parent.parent / "web" / "public" / "data" / "index.json"


def post_diagnose(base_url: str, case_id: str, model_id: str) -> dict:
    payload = json.dumps({"case_id": case_id, "model_id": model_id}).encode()
    req = urllib.request.Request(
        f"{base_url}/diagnose",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    elapsed = (time.perf_counter() - t0) * 1000
    return {"elapsed_ms": elapsed, "source": data.get("source"), "model": model_id, "case": case_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--case", default=None, help="Specific case_id (default: first from index)")
    args = parser.parse_args()

    # Load a case_id from index
    index = json.loads(INDEX_FILE.read_text())
    case_id = args.case or index["cases"][0]["id"]
    models = ["conflux", "quorum", "nexus"]

    print(f"Benchmark target: {args.url}")
    print(f"Using case: {case_id}")
    print()

    # ── Sequential: 50 calls ──────────────────────────────────────────────────
    print("=" * 60)
    print("SEQUENTIAL (50 calls):")
    latencies = []
    for i in range(50):
        mid = models[i % 3]
        res = post_diagnose(args.url, case_id, mid)
        latencies.append(res["elapsed_ms"])
        print(f"  [{i+1:02d}] {mid:8s} {res['elapsed_ms']:.0f} ms  source={res['source']}")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    print(f"\n  p50={p50:.0f} ms  p95={p95:.0f} ms  p99={p99:.0f} ms  avg={sum(latencies)/len(latencies):.0f} ms")
    print(f"  Budget: p95 ≤ 1500 ms → {'PASS ✓' if p95 <= 1500 else 'FAIL ✗'}")

    # ── Concurrent: 10 calls ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("CONCURRENT (10 simultaneous calls):")
    conc_results = []
    errors = []

    def _worker(idx):
        mid = models[idx % 3]
        try:
            res = post_diagnose(args.url, case_id, mid)
            conc_results.append(res["elapsed_ms"])
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - t_start) * 1000

    print(f"  Completed in {wall_ms:.0f} ms wall-clock, {len(conc_results)} ok, {len(errors)} errors")
    if conc_results:
        conc_results.sort()
        print(f"  Individual: min={conc_results[0]:.0f} ms  max={conc_results[-1]:.0f} ms")
    if errors:
        print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
