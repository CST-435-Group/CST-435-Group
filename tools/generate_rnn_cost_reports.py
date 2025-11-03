"""Generate per-project RNN deployment cost analysis reports.

This script scans top-level project folders and detects projects that look RNN-related
(either directory name contains 'rnn' or they contain model files like .pt/.pth).
For each detected project it generates a markdown report under docs/cost_reports.

The report uses the template provided by the user and fills detectable values:
- project name
- analysis date
- model file(s) and sizes
- dataset size (sum of files in data/ or backend/data/)
- architecture hints (searches README and model config JSONs)

Remaining items (pricing, instance types, hourly rates) are left as placeholders
so you can fill them with real cloud pricing or run a follow-up to estimate costs.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime
import re
from typing import Optional, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "docs" / "cost_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def find_model_files(project_path: Path) -> List[Path]:
    candidates = []
    # common model locations
    for p in project_path.rglob("*.pt"):
        candidates.append(p)
    for p in project_path.rglob("*.pth"):
        candidates.append(p)
    for p in project_path.rglob("*.bin"):
        candidates.append(p)
    return sorted(set(candidates))


def find_dataset_dirs(project_path: Path) -> List[Path]:
    paths = []
    for candidate in [project_path / "data", project_path / "backend" / "data", project_path / "datafiles"]:
        if candidate.exists():
            paths.append(candidate)
    return paths


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def read_architecture_hint(project_path: Path) -> Optional[str]:
    # search README or any config JSON for keywords
    keywords = ["LSTM", "GRU", "RNN", "lstm", "gru", "vanilla rnn", "rnn"]
    # README files
    for name in ("README.md", "README.MD", "README.txt", "README"):
        p = project_path / name
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                for kw in keywords:
                    if kw in text:
                        return kw.upper()
            except OSError:
                pass
    # look in model config json files
    for cfg in project_path.rglob("*_config.json"):
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            # common keys
            for key in ("architecture", "arch", "model", "type"):
                if key in data:
                    v = str(data[key])
                    for kw in keywords:
                        if kw.lower() in v.lower():
                            return kw.upper()
            # try to look for layer names
            content = json.dumps(data)
            for kw in keywords:
                if kw.lower() in content.lower():
                    return kw.upper()
        except Exception:
            continue
    return None


def try_read_parameters_from_config(project_path: Path) -> Optional[str]:
    for cfg in project_path.rglob("*_config.json"):
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            for key in ("parameters", "num_parameters", "n_params", "params"):
                if key in data:
                    return str(data[key])
            # maybe model size or counts present
            if "model_size" in data:
                return str(data["model_size"])
        except Exception:
            continue
    return None


def generate_report_for(project_path: Path) -> Path:
    project_name = project_path.name
    analysis_date = datetime.utcnow().strftime("%Y-%m-%d")

    model_files = find_model_files(project_path)
    model_info = []
    total_model_bytes = 0
    for m in model_files:
        try:
            sz = m.stat().st_size
        except OSError:
            sz = 0
        total_model_bytes += sz
        model_info.append({"file": str(m.relative_to(ROOT)), "size_bytes": sz, "size_human": human_size(sz)})

    dataset_dirs = find_dataset_dirs(project_path)
    dataset_info = []
    total_dataset_bytes = 0
    for d in dataset_dirs:
        ds = dir_size(d)
        total_dataset_bytes += ds
        dataset_info.append({"path": str(d.relative_to(ROOT)), "size_bytes": ds, "size_human": human_size(ds)})

    arch_hint = read_architecture_hint(project_path) or "(unknown - please fill)"
    params_hint = try_read_parameters_from_config(project_path) or "(unknown - please fill)"

    # find any config JSON near model files
    config_paths = [str(p.relative_to(ROOT)) for p in project_path.rglob("*_config.json")]

    md_lines = []
    md_lines.append("===== RNN DEPLOYMENT COST ANALYSIS REPORT =====")
    md_lines.append("")
    md_lines.append(f"Project: {project_name}")
    md_lines.append("Cloud Provider: [AWS/Azure/GCP/Other]")
    md_lines.append("Region: [e.g., US-East]")
    md_lines.append(f"Analysis Date: {analysis_date}")
    md_lines.append("")
    md_lines.append("--- MODEL SPECIFICATIONS ---")
    md_lines.append(f"Architecture: {arch_hint}")
    md_lines.append(f"Parameters: {params_hint}")
    md_lines.append(f"Model Size: {human_size(total_model_bytes)}")
    if model_info:
        md_lines.append("")
        md_lines.append("Model files:")
        for mi in model_info:
            md_lines.append(f"- {mi['file']} — {mi['size_human']}")
    md_lines.append(f"Dataset Size: {human_size(total_dataset_bytes)}")
    if dataset_info:
        md_lines.append("")
        md_lines.append("Dataset sources:")
        for di in dataset_info:
            md_lines.append(f"- {di['path']} — {di['size_human']}")

    md_lines.append("")
    md_lines.append("--- INFRASTRUCTURE SPECIFICATIONS ---")
    md_lines.append("Instance Type: [e.g., c5.2xlarge]")
    md_lines.append("vCPUs: [e.g., 8 cores]")
    md_lines.append("RAM: [e.g., 16 GB]")
    md_lines.append("Storage: [e.g., 50 GB SSD]")
    md_lines.append("")
    md_lines.append("--- COST BREAKDOWN ---")
    md_lines.append("")
    md_lines.append("1. TRAINING COSTS (One-time)")
    md_lines.append("   Compute: $XX.XX ([X] hours × $Y.YY/hour)")
    md_lines.append("   Storage: $XX.XX ([X] GB × $Y.YY/GB)")
    md_lines.append("   Data Transfer: $XX.XX")
    md_lines.append("   Total Training Cost: $XXX.XX")
    md_lines.append("")
    md_lines.append("2. INFERENCE COSTS (Monthly)")
    md_lines.append("   Compute: $XX.XX ([X] hours × $Y.YY/hour)")
    md_lines.append("   Storage: $XX.XX")
    md_lines.append("   Data Transfer: $XX.XX ([X] GB × $Y.YY/GB)")
    md_lines.append("   Total Monthly Cost: $XXX.XX")
    md_lines.append("")
    md_lines.append("3. KEY METRICS")
    md_lines.append("   Cost per Inference: $X.XXXX")
    md_lines.append("   Inferences per Dollar: XXX")
    md_lines.append("   Monthly Inference Capacity: XXX,XXX requests")
    md_lines.append("")
    md_lines.append("4. SCALING SCENARIOS")
    md_lines.append("   Low Volume (100/day): $XX.XX/month")
    md_lines.append("   Medium Volume (10,000/day): $XX.XX/month")
    md_lines.append("   High Volume (1M/day): $XX.XX/month")
    md_lines.append("")
    md_lines.append("--- ASSUMPTIONS ---")
    md_lines.append("• [List all assumptions made, e.g., \"Assumed 99% uptime\"]")
    md_lines.append("• [e.g., \"Used on-demand pricing; reserved instances would reduce cost by ~40%\"]")
    md_lines.append("• [e.g., \"Assumed average inference time of 50ms based on local testing\"]")
    md_lines.append("• [e.g., \"Data transfer costs assume 10 KB per request\"]")
    md_lines.append("")
    md_lines.append("--- OPTIMIZATION RECOMMENDATIONS ---")
    md_lines.append("• [e.g., \"Consider spot instances for training to reduce costs by 70%\"]")
    md_lines.append("• [e.g., \"Implement model quantization to reduce inference cost\"]")
    md_lines.append("• [e.g., \"Use caching for repeated requests\"]")
    md_lines.append("")
    md_lines.append("--- COST COMPARISON ---")
    md_lines.append("[Compare with alternative approaches if applicable]")
    md_lines.append("Alternative Model: [e.g., Transformer model]")
    md_lines.append("Cost Difference: [e.g., \"+150% more expensive\"]")
    md_lines.append("Performance Difference: [e.g., \"+10% accuracy\"]")
    md_lines.append("Cost-Efficiency Trade-off: [Analysis]")
    md_lines.append("")
    md_lines.append("==========================================")
    md_lines.append("")
    if config_paths:
        md_lines.append("Detected config files (check for parameters/architecture):")
        for cp in config_paths:
            md_lines.append(f"- {cp}")
        md_lines.append("")

    out_path = REPORTS_DIR / f"{project_name}_rnn_deployment_cost_report.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    return out_path


def fill_defaults_in_report(report_path: Path, model_bytes: int, dataset_bytes: int):
    """Fill cost placeholders in an existing report using recommended defaults (AWS us-east-1).

    This uses simple, transparent assumptions; numbers are illustrative and should be
    adjusted with real cloud pricing for production decisions.
    """
    # defaults (recommendations)
    defaults = {
        "provider": "AWS",
        "region": "us-east-1",
        "training_instance": {"name": "g4dn.xlarge", "vcpus": 4, "hourly": 0.526},
        "inference_instance": {"name": "t3.medium", "vcpus": 2, "hourly": 0.0416},
        "storage_gb_month": 0.08,  # $/GB-month (EBS gp3-like)
        "data_transfer_gb": 0.09,  # $/GB out
        "training_hours": 20,      # default training runtime
        "inference_time_ms": 50,   # average inference time
        "data_per_request_kb": 10, # payload per request
        "utilization": 0.7,
    }

    model_gb = model_bytes / (1024 ** 3)
    dataset_gb = dataset_bytes / (1024 ** 3)

    # Training costs
    train_hours = defaults["training_hours"]
    train_hourly = defaults["training_instance"]["hourly"]
    training_compute = train_hours * train_hourly
    training_storage = (dataset_gb + model_gb) * defaults["storage_gb_month"]
    training_data_transfer = dataset_gb * defaults["data_transfer_gb"]
    total_training = training_compute + training_storage + training_data_transfer

    # Inference monthly costs
    inf_hourly = defaults["inference_instance"]["hourly"]
    vcpus = defaults["inference_instance"]["vcpus"]
    inf_time_s = defaults["inference_time_ms"] / 1000.0
    inf_per_sec_per_core = 1.0 / inf_time_s if inf_time_s > 0 else 0
    inf_per_sec_instance = vcpus * inf_per_sec_per_core * defaults["utilization"]
    seconds_per_month = 30 * 24 * 3600
    monthly_capacity = inf_per_sec_instance * seconds_per_month

    cost_per_inference = (inf_hourly) / (inf_per_sec_instance * 3600) if inf_per_sec_instance > 0 else float("inf")
    inferences_per_dollar = 1.0 / cost_per_inference if cost_per_inference > 0 and cost_per_inference != float("inf") else 0

    def monthly_cost_for_requests(requests_per_day: int):
        requests_month = requests_per_day * 30
        instances_needed = max(1, int((requests_month / monthly_capacity) + (0 if requests_month % monthly_capacity == 0 else 1)))
        compute = instances_needed * inf_hourly * 24 * 30
        storage = (model_gb) * defaults["storage_gb_month"]
        data_transfer_gb_month = requests_month * (defaults["data_per_request_kb"] / 1024.0 / 1024.0)
        data_transfer_cost = data_transfer_gb_month * defaults["data_transfer_gb"]
        total = compute + storage + data_transfer_cost
        return {
            "instances": instances_needed,
            "compute": compute,
            "storage": storage,
            "data_transfer": data_transfer_cost,
            "total": total,
            "requests_month": requests_month,
        }

    low = monthly_cost_for_requests(100)
    medium = monthly_cost_for_requests(10_000)
    high = monthly_cost_for_requests(1_000_000)

    # Read original report and replace placeholders
    text = report_path.read_text(encoding="utf-8")
    # Replace training section
    text = re.sub(r"Compute: \$XX\.XX \(\[X\] hours × \$Y\.YY/hour\)",
                  f"Compute: ${training_compute:.2f} ({train_hours} hours × ${train_hourly:.3f}/hour)", text)
    text = re.sub(r"Storage: \$XX\.XX \(\[X\] GB × \$Y\.YY/GB\)",
                  f"Storage: ${training_storage:.2f} ({(dataset_gb+model_gb):.2f} GB × ${defaults['storage_gb_month']:.2f}/GB-month)", text)
    text = re.sub(r"Data Transfer: \$XX\.XX", f"Data Transfer: ${training_data_transfer:.2f}", text)
    text = re.sub(r"Total Training Cost: \$XXX\.XX", f"Total Training Cost: ${total_training:.2f}", text)

    # Inference
    text = re.sub(r"Compute: \$XX\.XX \(\[X\] hours × \$Y\.YY/hour\)",
                  f"Compute: ${inf_hourly:.2f} ({inf_hourly:.3f}/hour instance '{defaults['inference_instance']['name']}')", text, count=1)
    text = re.sub(r"Storage: \$XX\.XX", f"Storage: ${model_gb*defaults['storage_gb_month']:.2f}", text, count=1)
    text = re.sub(r"Data Transfer: \$XX\.XX \(\[X\] GB × \$Y\.YY/GB\)",
                  f"Data Transfer: ${((defaults['data_per_request_kb']/1024.0/1024.0) * 30*10000 * defaults['data_transfer_gb']):.2f} (example)", text, count=1)
    text = re.sub(r"Total Monthly Cost: \$XXX\.XX", f"Total Monthly Cost: ${medium['total']:.2f} (example for 10k/day)", text, count=1)

    # Key metrics
    text = re.sub(r"Cost per Inference: \$X\.XXXX", f"Cost per Inference: ${cost_per_inference:.6f}", text)
    text = re.sub(r"Inferences per Dollar: XXX", f"Inferences per Dollar: {int(inferences_per_dollar):,}", text)
    text = re.sub(r"Monthly Inference Capacity: XXX,XXX requests", f"Monthly Inference Capacity: {int(monthly_capacity):,} requests", text)

    # Scaling scenarios
    text = re.sub(r"Low Volume \(100/day\): \$XX\.XX/month", f"Low Volume (100/day): ${low['total']:.2f}/month", text)
    text = re.sub(r"Medium Volume \(10,000/day\): \$XX\.XX/month", f"Medium Volume (10,000/day): ${medium['total']:.2f}/month", text)
    text = re.sub(r"High Volume \(1M/day\): \$XX\.XX/month", f"High Volume (1M/day): ${high['total']:.2f}/month", text)

    # Assumptions section - append explicit defaults used
    assumptions = [
        f"• Provider default: {defaults['provider']} {defaults['region']}",
        f"• Training instance: {defaults['training_instance']['name']} at ${defaults['training_instance']['hourly']}/hour",
        f"• Inference instance: {defaults['inference_instance']['name']} at ${defaults['inference_instance']['hourly']}/hour",
        f"• Storage rate: ${defaults['storage_gb_month']}/GB-month",
        f"• Data transfer rate: ${defaults['data_transfer_gb']}/GB",
        f"• Assumed training time: {train_hours} hours",
        f"• Assumed average inference time: {defaults['inference_time_ms']} ms",
        f"• Assumed data per request: {defaults['data_per_request_kb']} KB",
    ]
    text = text + "\n--- Computation assumptions used to fill defaults ---\n" + "\n".join(assumptions) + "\n"

    report_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate RNN deployment cost reports for selected projects or auto-detect.")
    parser.add_argument("--projects", "-p", nargs="*", help="Project folder names to generate reports for (default: auto-detect)")
    parser.add_argument("--fill-defaults", action="store_true", help="Fill cost placeholders using recommended defaults (AWS us-east-1)")
    args = parser.parse_args()

    # detect candidate projects at top level
    candidates = []
    if args.projects:
        for name in args.projects:
            p = ROOT / name
            if p.exists() and p.is_dir():
                candidates.append(p)
            else:
                print(f"Warning: requested project '{name}' not found under {ROOT}")
    else:
        for p in ROOT.iterdir():
            if not p.is_dir():
                continue
            name_lower = p.name.lower()
            # heuristics
            if "rnn" in name_lower or "recurrent" in name_lower:
                candidates.append(p)
                continue
            # folders with saved_models or models
            if (p / "backend" / "saved_models").exists():
                candidates.append(p)
                continue
            if (p / "models").exists():
                # ensure there are model files
                mfs = list((p / "models").glob("*.pth")) + list((p / "models").glob("*.pt"))
                if mfs:
                    candidates.append(p)
                    continue
        # also check for app-level routers mentioning rnn (helpful for monorepos)
        for p in ROOT.rglob("rnn.py"):
            extra_proj = p.parents[2] if len(p.parents) > 2 else p.parents[0]
            if extra_proj.is_dir() and extra_proj not in candidates:
                candidates.append(extra_proj)

    if not candidates:
        print("No candidate RNN projects detected. Nothing to do.")
        raise SystemExit(0)

    print(f"Detected {len(candidates)} candidate(s): {[c.name for c in candidates]}")
    generated = []
    for c in sorted(candidates, key=lambda x: x.name):
        print(f"Processing {c.name} ...")
        out = generate_report_for(c)
        generated.append(out)
        print(f" -> wrote {out}")
        if args.fill_defaults:
            # recompute sizes
            mfs = find_model_files(c)
            total_model_bytes = sum((p.stat().st_size for p in mfs), 0)
            dpaths = find_dataset_dirs(c)
            total_dataset_bytes = sum((dir_size(d) for d in dpaths), 0)
            try:
                fill_defaults_in_report(out, total_model_bytes, total_dataset_bytes)
                print(f" -> filled defaults into {out}")
            except Exception as e:
                print(f" -> failed filling defaults for {c.name}: {e}")

    print("\nDone. Generated reports:")
    for g in generated:
        print(f" - {g}")
    print(f"Reports are in {REPORTS_DIR}")
