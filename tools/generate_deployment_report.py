"""Generate a human-readable deployment cost analysis markdown report for a project.

The script prefers an existing `docs/cost_report.json` for the project. If absent,
it falls back to using `cost_analysis.compute_phased_costs()` or `example_scenario()`.

Usage:
  python tools/generate_deployment_report.py --project rnn --provider AWS --region us-east-1 --project-name "RNN Text Generator"
"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime


PROJECT_MAP = {
    'ann': 'ANN_Project',
    'cnn': 'CNN_Project',
    'nlp': 'NLP',
    'rnn': 'rnn-text-generator'
}


def load_project_cost(project_dir: Path):
    json_path = project_dir / 'docs' / 'cost_report.json'
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def human_fmt_number(x):
    try:
        if isinstance(x, int):
            return f"{x:,}"
        return f"{x:,.4f}" if x < 1 else f"{x:,.2f}"
    except Exception:
        return str(x)


def build_markdown(project_name, provider, region, analysis_date, specs, infra, phase1, phase2, assumptions, scaling):
    lines = []
    lines.append("===== RNN DEPLOYMENT COST ANALYSIS REPORT =====\n")
    lines.append(f"Project: {project_name}")
    lines.append(f"Cloud Provider: {provider}")
    lines.append(f"Region: {region}")
    lines.append(f"Analysis Date: {analysis_date}\n")

    lines.append("--- MODEL SPECIFICATIONS ---")
    lines.append(f"Architecture: {specs.get('architecture', 'RNN')}")
    lines.append(f"Parameters: {specs.get('parameters', '')}")
    lines.append(f"Model Size: {specs.get('model_size', '')}")
    lines.append(f"Dataset Size: {specs.get('dataset_size', '')}\n")

    lines.append("--- INFRASTRUCTURE SPECIFICATIONS ---")
    lines.append(f"Instance Type: {infra.get('instance_type', '')}")
    lines.append(f"vCPUs: {infra.get('vcpus', '')}")
    lines.append(f"RAM: {infra.get('ram', '')}")
    lines.append(f"Storage: {infra.get('storage', '')}\n")

    lines.append("--- COST BREAKDOWN ---\n")
    lines.append("1. TRAINING COSTS (One-time)")
    lines.append(f"   Compute: ${human_fmt_number(phase1.get('compute_training', 0))} ({human_fmt_number(phase1.get('compute_training', 0)/phase1.get('compute_training',1) if phase1.get('compute_training') else 0)} hours × ${human_fmt_number(specs.get('train_cost_per_hour',''))}/hour)")
    lines.append(f"   Storage: ${human_fmt_number(phase1.get('storage_training', 0))} ({specs.get('dataset_size','')} × ${specs.get('storage_per_gb','')}/GB)")
    lines.append(f"   Data Transfer: ${human_fmt_number(phase1.get('data_transfer_training', 0))}")
    lines.append(f"   Total Training Cost: ${human_fmt_number(phase1.get('phase1_total', 0))}\n")

    lines.append("2. INFERENCE COSTS (Monthly)")
    lines.append(f"   Compute: ${human_fmt_number(phase2.get('compute_monthly', 0))} ({human_fmt_number(phase2.get('runtime_hours_month',0))} hours × ${human_fmt_number(infra.get('hourly_cost',''))}/hour)")
    lines.append(f"   Storage: ${human_fmt_number(phase2.get('storage_monthly', 0))}")
    lines.append(f"   Data Transfer: ${human_fmt_number(phase2.get('data_transfer_monthly', 0))} ({human_fmt_number(phase2.get('data_transfer_monthly',0)/ (phase2.get('requests_per_month',1)) if phase2.get('requests_per_month') else 0)} per request)")
    lines.append(f"   Total Monthly Cost: ${human_fmt_number(phase2.get('phase2_monthly_total', 0))}\n")

    lines.append("3. KEY METRICS")
    cost_per_inference = phase2.get('cost_per_request', 0)
    inferences_per_dollar = (1.0 / cost_per_inference) if cost_per_inference else None
    lines.append(f"   Cost per Inference: ${human_fmt_number(cost_per_inference)}")
    lines.append(f"   Inferences per Dollar: {human_fmt_number(inferences_per_dollar) if inferences_per_dollar else 'N/A'}")
    lines.append(f"   Monthly Inference Capacity: {human_fmt_number(phase2.get('requests_per_month',0))} requests\n")

    lines.append("4. SCALING SCENARIOS")
    for label, monthly in scaling.items():
        lines.append(f"   {label}: ${human_fmt_number(monthly)} /month")

    lines.append('\n--- ASSUMPTIONS ---')
    for a in assumptions:
        lines.append(f"• {a}")

    lines.append('\n--- OPTIMIZATION RECOMMENDATIONS ---')
    lines.append('• Consider spot instances for training to reduce costs by up to 70%')
    lines.append('• Implement model quantization to reduce inference compute and memory')
    lines.append('• Use caching for repeated requests and batching to improve throughput')

    lines.append('\n--- COST COMPARISON ---')
    lines.append('Alternative Model: Transformer (example)')
    lines.append('Cost Difference: +150% more expensive')
    lines.append('Performance Difference: +10% accuracy (example)')
    lines.append('Cost-Efficiency Trade-off: Transformers increase latency and cost; choose only if performance gain justifies spend')

    lines.append('\n==========================================')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='rnn', help='project id: ann, cnn, nlp, rnn')
    parser.add_argument('--provider', default='AWS')
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--project-name', default='RNN Project')
    parser.add_argument('--analysis-date', default=datetime.utcnow().date().isoformat())
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    if args.project not in PROJECT_MAP:
        print(f"Unknown project id: {args.project}")
        return

    proj_dir = repo_root / PROJECT_MAP[args.project]
    if not proj_dir.exists():
        print(f"Project directory not found: {proj_dir}")
        return

    # try to load existing cost JSON
    cost_json = load_project_cost(proj_dir)

    # import cost_analysis helpers for fallback or detailed calculations
    sys.path.insert(0, str(repo_root))
    from cost_analysis import compute_phased_costs, Pricing

    # Default specs & infra (can be improved to read per-project metadata)
    specs = {
        'architecture': 'LSTM',
        'parameters': '2.5M',
        'model_size': '10 MB',
        'dataset_size': '5 GB',
        'train_cost_per_hour': 0.5,
        'storage_per_gb': 0.02,
    }

    infra = {
        'instance_type': 'c5.2xlarge',
        'vcpus': '8',
        'ram': '16 GB',
        'storage': '50 GB SSD',
        'hourly_cost': 0.5,
    }

    pricing = Pricing(compute_per_hour=infra['hourly_cost'], storage_per_gb_month=specs['storage_per_gb'], data_transfer_per_gb=0.09, misc_monthly=10.0)

    if cost_json:
        # derive phased fields from existing cost JSON where possible
        # try to map fields
        phase1 = {
            'compute_training': cost_json.get('training_cost', 0),
            'storage_training': 0,
            'data_transfer_training': 0,
            'phase1_total': cost_json.get('training_cost', 0),
        }
        breakdown = cost_json.get('breakdown', {})
        infer_details = breakdown.get('monthly_inference_details', {})
        phase2 = {
            'runtime_hours_month': infer_details.get('compute_hours', 0),
            'compute_monthly': infer_details.get('compute_cost', 0),
            'cost_per_request': (infer_details.get('compute_cost',0) / (infer_details.get('requests_per_day',1)*30)) if infer_details.get('requests_per_day') else 0,
            'requests_per_month': infer_details.get('requests_per_day',0)*30 if infer_details.get('requests_per_day') else 0,
            'request_based_cost_monthly': cost_json.get('monthly_inference_cost', 0),
            'storage_monthly': cost_json.get('storage_cost_total', 0) / max(1, 6),
            'data_transfer_monthly': cost_json.get('monthly_data_transfer_cost', 0),
            'monthly_misc': cost_json.get('monthly_misc', 0),
            'phase2_monthly_total': cost_json.get('monthly_inference_cost', 0) + cost_json.get('monthly_data_transfer_cost', 0) + cost_json.get('monthly_misc', 0),
        }
    else:
        # fallback: use compute_phased_costs with example defaults
        phased = compute_phased_costs(
            training_hours=10.0,
            training_compute_cost_per_hour=pricing.compute_per_hour,
            dataset_size_gb=5.0,
            training_data_transfer_gb=2.0,
            inference_requests_per_day=1000,
            avg_latency_seconds=0.2,
            compute_cost_per_hour_inference=pricing.compute_per_hour,
            model_size_gb=0.2,
            deployment_months=6,
            pricing=pricing,
            avg_request_payload_gb=0.0001,
            avg_response_payload_gb=0.0005,
        )
        phase1 = phased['phase1']
        phase2 = phased['phase2_monthly']

    # assumptions
    assumptions = [
        'Assumed 99% uptime',
        'Used on-demand pricing; reserved instances would reduce cost by ~40%',
        'Assumed average inference time of 200ms (0.2s)',
        'Data transfer costs assume 0.6 KB per request (request+response)'
    ]

    # scaling scenarios: compute monthly totals for different request volumes
    scaling_inputs = {
        'Low Volume (100/day)': 100,
        'Medium Volume (10,000/day)': 10000,
        'High Volume (1M/day)': 1000000,
    }
    scaling = {}
    for label, rpd in scaling_inputs.items():
        ph = compute_phased_costs(
            training_hours=10.0,
            training_compute_cost_per_hour=pricing.compute_per_hour,
            dataset_size_gb=5.0,
            training_data_transfer_gb=2.0,
            inference_requests_per_day=rpd,
            avg_latency_seconds=0.2,
            compute_cost_per_hour_inference=pricing.compute_per_hour,
            model_size_gb=0.2,
            deployment_months=6,
            pricing=pricing,
            avg_request_payload_gb=0.0001,
            avg_response_payload_gb=0.0005,
        )
        scaling[label] = ph['phase2_monthly']['phase2_monthly_total']

    md = build_markdown(args.project_name, args.provider, args.region, args.analysis_date, specs, infra, phase1, phase2, assumptions, scaling)

    out_path = proj_dir / 'docs' / 'deployment_cost_report.md'
    out_path.write_text(md, encoding='utf-8')
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
