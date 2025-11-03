"""Simple inference benchmarking tool for the four projects.

Usage:
  python tools/run_inference_bench.py --base-url http://localhost:8000 --requests 30

Per-project defaults (can be changed in the PROJECTS dict):
- ann: POST /ann/select-team  (empty/default JSON)
- cnn: POST /cnn/predict      (multipart file; script creates a tiny grayscale image)
- nlp: POST /nlp/analyze     (JSON with a short text)
- rnn: POST /rnn/generate    (JSON sampling request)

The script performs a short warmup (3 requests), then N measured requests sequentially.
It writes results to docs/cost_reports/bench_{project}.json and prints summary to stdout.
"""
from __future__ import annotations
import requests
import time
import statistics
import json
import io
from PIL import Image
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "docs" / "cost_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS = {
    "ann": {
        "endpoint": "/ann/select-team",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json": {},
        "note": "Uses default TeamSelectionRequest payload"
    },
    "cnn": {
        "endpoint": "/cnn/predict",
        "method": "POST",
        "files": True,
        "note": "Sends a tiny generated image as multipart file"
    },
    "nlp": {
        "endpoint": "/nlp/analyze",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json": {"text": "This is a short test review used for benchmarking inference latency."},
        "note": "Single text sentiment analysis"
    },
    "rnn": {
        "endpoint": "/rnn/generate",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "json": {"seed_text": "Hello world", "num_words": 20, "temperature": 1.0, "use_beam_search": False},
        "note": "Text generation default request"
    }
}


def make_sample_image_bytes():
    # Create a small grayscale 128x128 image to match CNN preprocess
    img = Image.new('RGB', (128, 128), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def bench_request(session: requests.Session, method: str, url: str, proj_cfg: dict):
    start = time.perf_counter()
    if proj_cfg.get('files'):
        # send generated image
        img_buf = make_sample_image_bytes()
        files = {'file': ('bench.png', img_buf, 'image/png')}
        resp = session.post(url, files=files, timeout=30)
    else:
        headers = proj_cfg.get('headers')
        json_payload = proj_cfg.get('json')
        resp = session.request(method, url, json=json_payload, headers=headers, timeout=30)
    elapsed = time.perf_counter() - start
    return resp, elapsed


def run_bench(base_url: str, project_id: str, n_requests: int = 30, warmup: int = 3):
    if project_id not in PROJECTS:
        raise ValueError(f"Unknown project id: {project_id}")
    cfg = PROJECTS[project_id]
    url = base_url.rstrip('/') + cfg['endpoint']
    session = requests.Session()

    print(f"Benchmarking {project_id} -> {url}")
    print(f"Note: {cfg.get('note')}")

    # Warmup
    print(f"Warming up ({warmup} requests)...")
    for i in range(warmup):
        try:
            resp, t = bench_request(session, cfg.get('method', 'GET'), url, cfg)
            print(f" warmup #{i+1}: status={getattr(resp, 'status_code', 'ERR')} time={t:.4f}s")
        except Exception as e:
            print(f" warmup #{i+1} failed: {e}")

    # Measured requests
    times = []
    statuses = []
    for i in range(n_requests):
        try:
            resp, t = bench_request(session, cfg.get('method', 'GET'), url, cfg)
            times.append(t)
            statuses.append(getattr(resp, 'status_code', None))
            print(f" #{i+1}: status={resp.status_code} time={t:.4f}s")
        except Exception as e:
            print(f" #{i+1}: failed: {e}")
            times.append(None)
            statuses.append(None)

    # Clean times (exclude None)
    times_clean = [t for t in times if t is not None]
    summary = {
        "project": project_id,
        "requests_measured": len(times_clean),
        "requests_total": n_requests,
        "status_codes": {str(code): statuses.count(code) for code in set(statuses) if code is not None},
        "mean_s": statistics.mean(times_clean) if times_clean else None,
        "median_s": statistics.median(times_clean) if times_clean else None,
        "p95_s": (sorted(times_clean)[int(len(times_clean)*0.95)-1] if times_clean and len(times_clean) >= 1 else None),
        "min_s": min(times_clean) if times_clean else None,
        "max_s": max(times_clean) if times_clean else None,
    }

    out_path = REPORTS_DIR / f"bench_{project_id}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote benchmark summary to {out_path}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference benchmark for projects')
    parser.add_argument('--base-url', default='http://localhost:8000', help='Base URL for API (default http://localhost:8000)')
    parser.add_argument('--projects', '-p', nargs='*', default=['ann', 'cnn', 'nlp', 'rnn'], help='Project ids to benchmark')
    parser.add_argument('--requests', '-n', type=int, default=30, help='Number of measured requests per project')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup requests')
    args = parser.parse_args()

    for pid in args.projects:
        try:
            s = run_bench(args.base_url, pid, n_requests=args.requests, warmup=args.warmup)
            print(json.dumps(s, indent=2))
        except Exception as e:
            print(f"Failed to benchmark {pid}: {e}")

    print("Done.")
