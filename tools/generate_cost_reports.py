"""Generate per-project cost JSON files using cost_analysis helpers.

Usage examples:
  # generate for all known projects
  python tools/generate_cost_reports.py --all

  # generate for specific projects
  python tools/generate_cost_reports.py --projects ann cnn

This script will write `docs/cost_report.json` under each project directory.
"""
import argparse
import json
from pathlib import Path
import sys


PROJECT_MAP = {
    'ann': 'ANN_Project',
    'cnn': 'CNN_Project',
    'nlp': 'NLP',
    'rnn': 'rnn-text-generator'
}


def main():
    parser = argparse.ArgumentParser(description="Generate cost_report.json files for projects")
    parser.add_argument('--projects', nargs='+', help='project ids to generate (ann, cnn, nlp, rnn)')
    parser.add_argument('--all', action='store_true', help='Generate for all projects')
    parser.add_argument('--use-phased', action='store_true', help='Use compute_phased_costs output instead of example_scenario')
    args = parser.parse_args()

    # Ensure repo root on sys.path so we can import cost_analysis
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    try:
        from cost_analysis import example_scenario, compute_phased_costs, Pricing
    except Exception as e:
        print(f"Unable to import cost_analysis: {e}")
        raise

    targets = []
    if args.all or not args.projects:
        targets = list(PROJECT_MAP.keys())
    else:
        targets = args.projects

    for pid in targets:
        if pid not in PROJECT_MAP:
            print(f"Unknown project id: {pid}, skipping")
            continue

        proj_dir = repo_root / PROJECT_MAP[pid]
        if not proj_dir.exists():
            print(f"Project directory not found: {proj_dir}, skipping")
            continue

        docs_dir = proj_dir / 'docs'
        docs_dir.mkdir(parents=True, exist_ok=True)

        if args.use_phased:
            # Use some reasonable defaults similar to example_scenario
            pricing = Pricing(0.5, 0.02, 0.09, misc_monthly=10.0)
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
            out = phased
        else:
            out = example_scenario()

        out_path = docs_dir / 'cost_report.json'
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

        print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
