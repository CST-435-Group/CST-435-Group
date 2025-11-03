"""Apply measured benchmark latencies to project cost JSONs.

Reads bench_{project}.json files in docs/cost_reports/ and updates each project's
`<project>/docs/cost_report.json` with computed monthly inference costs using the
measured median latency.

Assumptions (adjustable):
- inference instance: t3.medium (2 vCPUs) @ $0.0416/hr
- utilization: 0.7
- storage rate: $0.08 / GB-month
- data per request: 10 KB
- data transfer rate: $0.09 / GB
- monthly window: 30 days
- compute for training preserved from existing project cost JSON if present, else default 10.0
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "docs" / "cost_reports"

DEFAULTS = {
    "inference_instance_hourly": 0.0416,
    "inference_instance_vcpus": 2,
    "utilization": 0.7,
    "storage_gb_month": 0.08,
    "data_per_request_kb": 10,
    "data_transfer_gb_rate": 0.09,
    "monthly_days": 30
}

PROJECT_MAP = {
    'ann': ROOT / 'ANN_Project' / 'docs' / 'cost_report.json',
    'cnn': ROOT / 'CNN_Project' / 'docs' / 'cost_report.json',
    'nlp': ROOT / 'NLP' / 'docs' / 'cost_report.json',
    'rnn': ROOT / 'rnn-text-generator' / 'docs' / 'cost_report.json'
}

# Instance presets (hourly, vcpus)
INSTANCE_PRESETS = {
    't3.medium': {'hourly': 0.0416, 'vcpus': 2},
    'g4dn.xlarge': {'hourly': 0.526, 'vcpus': 4},
    # add more presets if needed
}

SCENARIOS = [10_000, 100_000, 1_000_000]


def read_json(p: Path):
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def write_json(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')


def apply_for_project(proj_id: str):
    bench_f = REPORTS_DIR / f'bench_{proj_id}.json'
    if not bench_f.exists():
        print(f'No bench file for {proj_id} at {bench_f}, skipping')
        return
    bench = read_json(bench_f)
    median_s = bench.get('median_s')
    if median_s is None:
        print(f'No median latency for {proj_id} in bench file, skipping')
        return

    # Determine project root for model size inspection
    proj_root = None
    if proj_id == 'ann':
        proj_root = ROOT / 'ANN_Project'
    elif proj_id == 'cnn':
        proj_root = ROOT / 'CNN_Project'
    elif proj_id == 'nlp':
        proj_root = ROOT / 'NLP'
    elif proj_id == 'rnn':
        proj_root = ROOT / 'rnn-text-generator'

    model_size_gb = 0.0
    if proj_root:
        models_dir = proj_root / 'models'
        model_paths = list(models_dir.rglob('*.pt')) if models_dir.exists() else []
        if not model_paths and (proj_root / 'backend' / 'saved_models').exists():
            model_paths = list((proj_root / 'backend' / 'saved_models').rglob('*.pt'))
        if model_paths:
            total_bytes = sum((p.stat().st_size for p in model_paths if p.exists()), 0)
            model_size_gb = total_bytes / (1024 ** 3)

    # Auto-select instance presets
    chosen_instance = 't3.medium'
    if proj_id == 'cnn':
        chosen_instance = 'g4dn.xlarge'
    elif proj_id == 'rnn':
        if model_size_gb > 0.05:
            chosen_instance = 'g4dn.xlarge'
        else:
            chosen_instance = 't3.medium'

    inst_info = INSTANCE_PRESETS.get(chosen_instance, {'hourly': DEFAULTS['inference_instance_hourly'], 'vcpus': DEFAULTS['inference_instance_vcpus']})
    inf_hourly = inst_info['hourly']
    vcpus = inst_info['vcpus']

    # compute throughput from median
    per_core_throughput = 1.0 / median_s if median_s > 0 else 0
    instance_throughput_per_sec = per_core_throughput * vcpus * DEFAULTS['utilization']
    seconds_per_month = DEFAULTS['monthly_days'] * 24 * 3600
    monthly_capacity = instance_throughput_per_sec * seconds_per_month
    cost_per_inference = inf_hourly / (instance_throughput_per_sec * 3600) if instance_throughput_per_sec > 0 else None

    # storage cost (model)
    storage_cost = model_size_gb * DEFAULTS['storage_gb_month']

    # build multi-scenario outputs
    scenarios_out = {}
    data_per_request_gb = DEFAULTS['data_per_request_kb'] / 1024.0 / 1024.0
    for req_per_day in SCENARIOS:
        requests_month = req_per_day * DEFAULTS['monthly_days']
        instances_needed = max(1, math.ceil(requests_month / monthly_capacity)) if monthly_capacity > 0 else 1
        compute_monthly = instances_needed * inf_hourly * 24 * DEFAULTS['monthly_days']
        data_transfer_gb_month = requests_month * data_per_request_gb
        data_transfer_cost = data_transfer_gb_month * DEFAULTS['data_transfer_gb_rate']
        total = compute_monthly + storage_cost + data_transfer_cost
        scenarios_out[f'cost_{req_per_day}_per_day'] = round(total, 2)
        scenarios_out[f'instances_{req_per_day}_per_day'] = instances_needed

    # attempt to read existing project cost report to get training cost
    proj_cost_path = PROJECT_MAP.get(proj_id)
    existing = read_json(proj_cost_path) if proj_cost_path else {}
    training_cost = existing.get('training_cost', 10.0)

    out = {
        'project': proj_id,
        'chosen_instance': chosen_instance,
        'median_inference_s': median_s,
        'instance_throughput_per_sec': instance_throughput_per_sec,
        'monthly_capacity': monthly_capacity,
        'cost_per_inference': cost_per_inference,
        'model_size_gb': round(model_size_gb, 4),
        'training_cost': training_cost,
        'storage_cost': round(storage_cost, 4),
    }
    out.update(scenarios_out)
    # Backwards-compatible fields expected by the frontend (use 10k/day as the default scenario)
    sample_requests = SCENARIOS[0]
    sample_total = scenarios_out.get(f'cost_{sample_requests}_per_day', 0)
    sample_instances = scenarios_out.get(f'instances_{sample_requests}_per_day', 1)
    # compute_monthly for the sample is total minus storage and data_transfer
    # reconstruct data_transfer and compute_monthly for the 10k/day sample
    requests_month_sample = sample_requests * DEFAULTS['monthly_days']
    data_transfer_gb_month_sample = requests_month_sample * (DEFAULTS['data_per_request_kb'] / 1024.0 / 1024.0)
    data_transfer_cost_sample = data_transfer_gb_month_sample * DEFAULTS['data_transfer_gb_rate']
    compute_monthly_sample = sample_total - storage_cost - data_transfer_cost_sample

    out['monthly_inference_cost'] = round(max(0, compute_monthly_sample), 2)
    out['monthly_data_transfer_cost'] = round(data_transfer_cost_sample, 2)
    out['monthly_misc'] = 0.0
    out['total_over_deployment'] = round(training_cost + sample_total, 2)

    if proj_cost_path:
        write_json(proj_cost_path, out)
        print(f'Wrote updated cost JSON to {proj_cost_path}')
    else:
        print(f'No project cost path mapped for {proj_id}, would have produced:')
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    for pid in ['ann', 'cnn', 'nlp', 'rnn']:
        apply_for_project(pid)
    print('Done applying benchmark results to project cost JSONs')
