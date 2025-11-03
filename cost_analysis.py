"""
cost_analysis.py

Utility to estimate costs for ML training, inference and deployment.

Features:
- Training cost calculation
- Monthly inference cost calculation (based on requests, latency, and instance pricing)
- Storage and data transfer cost calculations
- Total cost of ownership (TCO)
- Sensitivity analysis for request volumes and instance types

This file is intentionally self-contained and does not call cloud APIs.
Usage: import the functions or run `python cost_analysis.py` to see the example scenario.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import math


@dataclass
class Pricing:
    compute_per_hour: float  # $ per hour for compute (training or inference instance)
    storage_per_gb_month: float  # $ per GB per month
    data_transfer_per_gb: float  # $ per GB transferred
    misc_monthly: float = 0.0  # other monthly recurring costs (monitoring, load balancer, etc.)
    ingress_per_gb: float = 0.0  # optional separate ingress price
    egress_per_gb: float = 0.0


def training_cost(training_hours: float, compute_cost_per_hour: float, additional_training_costs: float = 0.0) -> float:
    """Estimate training cost.

    training_hours: total hours spent training (including hyperparameter search)
    compute_cost_per_hour: price per compute hour
    additional_training_costs: e.g., spot instance overhead, storage for checkpoints, data egress
    """
    if training_hours < 0:
        raise ValueError("training_hours must be non-negative")
    base = training_hours * compute_cost_per_hour
    return base + additional_training_costs


def development_cost(dev_hours: float, dev_hourly_rate: float, tooling_costs: float = 0.0) -> float:
    """Estimate development (engineering) costs for model development and experiments."""
    if dev_hours < 0 or dev_hourly_rate < 0:
        raise ValueError("dev_hours and dev_hourly_rate must be non-negative")
    return dev_hours * dev_hourly_rate + tooling_costs


def storage_cost(size_gb: float, storage_cost_per_gb_month: float, months: float = 1.0) -> float:
    """Estimate storage cost for given size and duration (months).
    """
    if size_gb < 0 or months < 0:
        raise ValueError("size_gb and months must be non-negative")
    return size_gb * storage_cost_per_gb_month * months


def storage_over_time(size_gb: float, storage_cost_per_gb_month: float, months: int) -> Dict[int, float]:
    """Return storage cost per month and cumulative total for 'months' months.
    Returns a dict: {'monthly': [m1, m2, ...], 'total': total}
    """
    if months < 0:
        raise ValueError("months must be non-negative")
    monthly = [size_gb * storage_cost_per_gb_month for _ in range(months)]
    total = sum(monthly)
    return {"monthly": monthly, "total": total}


def data_transfer_cost(transfer_gb: float, data_transfer_cost_per_gb: float) -> float:
    if transfer_gb < 0:
        raise ValueError("transfer_gb must be non-negative")
    return transfer_gb * data_transfer_cost_per_gb


def data_ingress_egress_cost(ingress_gb: float, egress_gb: float, ingress_per_gb: float, egress_per_gb: float) -> float:
    """Compute separate ingress and egress costs."""
    if ingress_gb < 0 or egress_gb < 0:
        raise ValueError("ingress_gb and egress_gb must be non-negative")
    return ingress_gb * ingress_per_gb + egress_gb * egress_per_gb


def inference_monthly_cost(requests_per_day: float,
                           avg_latency_seconds: float,
                           compute_cost_per_hour_inference: float,
                           days: int = 30,
                           concurrency_factor: float = 1.0) -> Dict[str, float]:
    """Estimate monthly inference compute cost and approximate compute hours used.

    Approach:
    - Approximate total seconds of compute = requests_per_day * days * avg_latency_seconds * concurrency_factor
    - Convert seconds to hours and multiply by compute_cost_per_hour_inference

    concurrency_factor: allows modeling of parallelism/overhead (>1 increases compute time to account for inefficiencies)
    """
    if requests_per_day < 0 or avg_latency_seconds < 0 or days <= 0:
        raise ValueError("requests_per_day, avg_latency_seconds must be non-negative and days > 0")

    total_seconds = requests_per_day * days * avg_latency_seconds * concurrency_factor
    hours = total_seconds / 3600.0
    cost = hours * compute_cost_per_hour_inference
    return {
        "requests_per_day": requests_per_day,
        "days": days,
        "avg_latency_seconds": avg_latency_seconds,
        "compute_hours": hours,
        "compute_cost": cost
    }


def compare_retention_vs_scaling(baseline_monthly_compute_cost: float,
                                 deployment_months: int,
                                 retention_idle_monthly_cost: float = 0.0,
                                 scale_down_savings_fraction: float = 0.5,
                                 scale_down_monthly_overhead: float = 0.0,
                                 reload_one_time_cost: float = 0.0,
                                 reloads_per_month: float = 0.0) -> Dict[str, float]:
    """Compare two strategies over deployment months:
    - retain: keep resources charged at baseline_monthly_compute_cost every month
    - scale: reduce compute by scale_down_savings_fraction (0-1) and pay overheads plus reload costs

    Returns dict with retain_total and scale_total and monthly breakdowns.
    """
    if baseline_monthly_compute_cost < 0 or deployment_months <= 0:
        raise ValueError("baseline_monthly_compute_cost must be non-negative and deployment_months > 0")

    retain_monthly = baseline_monthly_compute_cost
    retain_total = retain_monthly * deployment_months

    # If scaling down, assume monthly compute cost is reduced by savings fraction
    scaled_monthly_compute = baseline_monthly_compute_cost * (1.0 - scale_down_savings_fraction)
    scaled_total_compute = scaled_monthly_compute * deployment_months

    # Add overheads and reloads
    total_overhead = scale_down_monthly_overhead * deployment_months
    total_reload = reload_one_time_cost * reloads_per_month * deployment_months

    scale_total = scaled_total_compute + total_overhead + total_reload

    return {
        "retain_monthly": retain_monthly,
        "retain_total": retain_total,
        "scaled_monthly_compute": scaled_monthly_compute,
        "scale_total": scale_total,
        "details": {
            "deployment_months": deployment_months,
            "scale_down_savings_fraction": scale_down_savings_fraction,
            "scale_down_monthly_overhead": scale_down_monthly_overhead,
            "reload_one_time_cost": reload_one_time_cost,
            "reloads_per_month": reloads_per_month
        }
    }


def monthly_inference_data_transfer(requests_per_day: float, avg_request_payload_gb: float, avg_response_payload_gb: float, days: int = 30) -> float:
    """Estimate monthly data transferred (GB) for inference.
    Includes request and response sizes.
    """
    if requests_per_day < 0 or avg_request_payload_gb < 0 or avg_response_payload_gb < 0:
        raise ValueError("payload sizes and requests_per_day must be non-negative")
    total_gb = requests_per_day * days * (avg_request_payload_gb + avg_response_payload_gb)
    return total_gb


def total_cost_of_ownership(training_hours: float,
                            inference_requests_per_day: float,
                            avg_latency_seconds: float,
                            model_size_gb: float,
                            dataset_size_gb: float,
                            deployment_months: float,
                            pricing: Pricing,
                            avg_request_payload_gb: float = 0.0,
                            avg_response_payload_gb: float = 0.0,
                            additional_training_costs: float = 0.0,
                            other_one_time_costs: float = 0.0) -> Dict[str, float]:
    """Compute a breakdown of one-time and recurring costs and return totals.

    Returns a dict with fields: training_cost, storage_cost, data_transfer_cost, monthly_inference_cost,
    monthly_misc, total_first_month, total_over_deployment
    """
    # Training (one-time)
    training = training_cost(training_hours, pricing.compute_per_hour, additional_training_costs)

    # Storage for model + dataset over the deployment duration
    storage = storage_cost(model_size_gb + dataset_size_gb, pricing.storage_per_gb_month, deployment_months)

    # Inference compute monthly
    infer = inference_monthly_cost(inference_requests_per_day, avg_latency_seconds, pricing.compute_per_hour, days=30)
    monthly_inference_cost_val = infer["compute_cost"]

    # Data transfer monthly
    monthly_transfer_gb = monthly_inference_data_transfer(inference_requests_per_day, avg_request_payload_gb, avg_response_payload_gb, days=30)
    monthly_data_transfer_cost = data_transfer_cost(monthly_transfer_gb, pricing.data_transfer_per_gb)

    # Monthly misc services
    monthly_misc = pricing.misc_monthly

    # Totals
    total_first_month = training + storage_cost(model_size_gb + dataset_size_gb, pricing.storage_per_gb_month, 1.0) + monthly_inference_cost_val + monthly_data_transfer_cost + monthly_misc + other_one_time_costs

    total_over_deployment = training + storage + (monthly_inference_cost_val + monthly_data_transfer_cost + monthly_misc) * deployment_months + other_one_time_costs

    return {
        "training_cost": training,
        "storage_cost_total": storage,
        "monthly_inference_cost": monthly_inference_cost_val,
        "monthly_data_transfer_cost": monthly_data_transfer_cost,
        "monthly_misc": monthly_misc,
        "total_first_month": total_first_month,
        "total_over_deployment": total_over_deployment,
        "breakdown": {
            "monthly_inference_details": infer,
            "monthly_transfer_gb": monthly_transfer_gb
        }
    }


def sensitivity_by_requests(requests_per_day_list: List[float],
                            avg_latency_seconds: float,
                            pricing: Pricing,
                            days: int = 30,
                            other_params: dict = None) -> List[Dict]:
    """Run a sensitivity analysis over a list of requests_per_day values.

    other_params can include model_size_gb, dataset_size_gb, deployment_months, avg_request_payload_gb, avg_response_payload_gb, training_hours
    """
    if other_params is None:
        other_params = {}

    results = []
    for r in requests_per_day_list:
        tco = total_cost_of_ownership(
            training_hours=other_params.get("training_hours", 0.0),
            inference_requests_per_day=r,
            avg_latency_seconds=avg_latency_seconds,
            model_size_gb=other_params.get("model_size_gb", 0.1),
            dataset_size_gb=other_params.get("dataset_size_gb", 1.0),
            deployment_months=other_params.get("deployment_months", 1.0),
            pricing=pricing,
            avg_request_payload_gb=other_params.get("avg_request_payload_gb", 0.0),
            avg_response_payload_gb=other_params.get("avg_response_payload_gb", 0.0),
            additional_training_costs=other_params.get("additional_training_costs", 0.0),
            other_one_time_costs=other_params.get("other_one_time_costs", 0.0)
        )
        results.append({"requests_per_day": r, "total_over_deployment": tco["total_over_deployment"], "monthly_inference_cost": tco["monthly_inference_cost"], "monthly_data_transfer_cost": tco["monthly_data_transfer_cost"]})

    return results


def compare_instance_types(instance_pricing: Dict[str, float],
                           training_hours: float,
                           inference_requests_per_day: float,
                           avg_latency_seconds: float,
                           **kwargs) -> List[Dict]:
    """Compare total cost over deployment for different compute prices.

    instance_pricing: mapping instance_name -> compute_cost_per_hour
    kwargs are forwarded to total_cost_of_ownership except pricing
    """
    results = []
    for name, cost_per_hour in instance_pricing.items():
        pricing = Pricing(compute_per_hour=cost_per_hour,
                          storage_per_gb_month=kwargs.get("storage_per_gb_month", 0.02),
                          data_transfer_per_gb=kwargs.get("data_transfer_per_gb", 0.09),
                          misc_monthly=kwargs.get("misc_monthly", 0.0))

        tco = total_cost_of_ownership(
            training_hours=training_hours,
            inference_requests_per_day=inference_requests_per_day,
            avg_latency_seconds=avg_latency_seconds,
            pricing=pricing,
            model_size_gb=kwargs.get("model_size_gb", 0.1),
            dataset_size_gb=kwargs.get("dataset_size_gb", 1.0),
            deployment_months=kwargs.get("deployment_months", 1.0),
            avg_request_payload_gb=kwargs.get("avg_request_payload_gb", 0.0),
            avg_response_payload_gb=kwargs.get("avg_response_payload_gb", 0.0),
            additional_training_costs=kwargs.get("additional_training_costs", 0.0),
            other_one_time_costs=kwargs.get("other_one_time_costs", 0.0)
        )

        results.append({"instance": name, "compute_per_hour": cost_per_hour, "total_over_deployment": tco["total_over_deployment"], "monthly_inference_cost": tco["monthly_inference_cost"]})

    return results


def compute_phased_costs(
    training_hours: float,
    training_compute_cost_per_hour: float,
    dataset_size_gb: float,
    training_data_transfer_gb: float,
    inference_requests_per_day: float,
    avg_latency_seconds: float,
    compute_cost_per_hour_inference: float,
    model_size_gb: float,
    deployment_months: int,
    pricing: Pricing,
    avg_request_payload_gb: float = 0.0,
    avg_response_payload_gb: float = 0.0,
    days_per_month: int = 30,
    derive_cost_per_request: bool = True,
    cost_per_request_override: float = None,
    dataset_storage_count_as_one_time: bool = True,
    months_for_training_storage: int = 1,
):
    """Compute costs separated into Phase 1 (training) and Phase 2 (inference/deployment).

    Returns a dict: {"phase1": {...}, "phase2_monthly": {...}, "cost_per_request": ...}
    """
    # Phase 1: Training
    compute_training = training_hours * training_compute_cost_per_hour
    if dataset_storage_count_as_one_time:
        storage_training = storage_cost(dataset_size_gb, pricing.storage_per_gb_month, months_for_training_storage)
    else:
        storage_training = 0.0
    data_transfer_training = data_transfer_cost(training_data_transfer_gb, pricing.data_transfer_per_gb)
    phase1_total = compute_training + storage_training + data_transfer_training

    # Phase 2: Inference / Deployment (monthly)
    # Runtime hours derived from requests * latency
    total_seconds_month = inference_requests_per_day * days_per_month * avg_latency_seconds
    runtime_hours_month = total_seconds_month / 3600.0
    compute_monthly = runtime_hours_month * compute_cost_per_hour_inference

    # cost per request (derived or overridden)
    if cost_per_request_override is not None:
        cost_per_request = cost_per_request_override
    else:
        # compute-cost per request + transfer per request
        compute_cost_per_req = compute_cost_per_hour_inference * (avg_latency_seconds / 3600.0)
        transfer_cost_per_req = (avg_request_payload_gb + avg_response_payload_gb) * pricing.data_transfer_per_gb
        cost_per_request = compute_cost_per_req + transfer_cost_per_req

    requests_per_month = inference_requests_per_day * days_per_month
    request_based_cost_monthly = cost_per_request * requests_per_month

    storage_monthly = storage_cost(model_size_gb, pricing.storage_per_gb_month, 1.0)
    data_transfer_monthly = monthly_inference_data_transfer(inference_requests_per_day, avg_request_payload_gb, avg_response_payload_gb, days=days_per_month) * pricing.data_transfer_per_gb

    phase2_monthly_total = compute_monthly + request_based_cost_monthly + storage_monthly + data_transfer_monthly + pricing.misc_monthly

    return {
        "phase1": {
            "compute_training": compute_training,
            "storage_training": storage_training,
            "data_transfer_training": data_transfer_training,
            "phase1_total": phase1_total,
        },
        "phase2_monthly": {
            "runtime_hours_month": runtime_hours_month,
            "compute_monthly": compute_monthly,
            "cost_per_request": cost_per_request,
            "requests_per_month": requests_per_month,
            "request_based_cost_monthly": request_based_cost_monthly,
            "storage_monthly": storage_monthly,
            "data_transfer_monthly": data_transfer_monthly,
            "monthly_misc": pricing.misc_monthly,
            "phase2_monthly_total": phase2_monthly_total,
        },
    }


def format_deployment_report(
    project_name: str = "RNN Project",
    provider: str = "AWS",
    region: str = "US-East",
    analysis_date: str = None,
    specs: dict = None,
    infra: dict = None,
    pricing: Pricing = None,
    training_hours: float = 10.0,
    training_data_transfer_gb: float = 2.0,
    inference_requests_per_day: float = 1000,
    avg_latency_seconds: float = 0.2,
    model_size_gb: float = 0.2,
    dataset_size_gb: float = 5.0,
    deployment_months: int = 6,
    avg_request_payload_gb: float = 0.0001,
    avg_response_payload_gb: float = 0.0005,
    days_per_month: int = 30,
):
    """Return a deployment cost analysis report (markdown string) in the requested template.

    The function fills missing information with sensible defaults and uses compute_phased_costs()
    to calculate training / inference costs and scaling scenarios.
    """
    from datetime import date

    if analysis_date is None:
        analysis_date = date.today().isoformat()

    # Defaults for specs and infra
    if specs is None:
        specs = {
            "architecture": "LSTM",
            "parameters": "2.5M",
            "model_size": f"{model_size_gb*1024:.0f} KB" if model_size_gb < 1 else f"{model_size_gb:.2f} GB",
            "dataset_size": f"{dataset_size_gb:.2f} GB",
            "train_cost_per_hour": None,
            "storage_per_gb": None,
        }

    if infra is None:
        infra = {
            "instance_type": "c5.2xlarge",
            "vcpus": "8",
            "ram": "16 GB",
            "storage": "50 GB SSD",
            "hourly_cost": None,
        }

    # Build pricing if not provided
    if pricing is None:
        compute_hour = infra.get("hourly_cost") or 0.5
        storage_per_gb = specs.get("storage_per_gb") or 0.02
        pricing = Pricing(compute_per_hour=compute_hour, storage_per_gb_month=storage_per_gb, data_transfer_per_gb=0.09, misc_monthly=10.0)

    # Phase calculations
    phased = compute_phased_costs(
        training_hours=training_hours,
        training_compute_cost_per_hour=pricing.compute_per_hour,
        dataset_size_gb=dataset_size_gb,
        training_data_transfer_gb=training_data_transfer_gb,
        inference_requests_per_day=inference_requests_per_day,
        avg_latency_seconds=avg_latency_seconds,
        compute_cost_per_hour_inference=pricing.compute_per_hour,
        model_size_gb=model_size_gb,
        deployment_months=deployment_months,
        pricing=pricing,
        avg_request_payload_gb=avg_request_payload_gb,
        avg_response_payload_gb=avg_response_payload_gb,
        days_per_month=days_per_month,
    )

    phase1 = phased["phase1"]
    phase2 = phased["phase2_monthly"]

    # Key metrics
    cost_per_inference = phase2.get("cost_per_request", 0)
    inferences_per_dollar = (1.0 / cost_per_inference) if cost_per_inference else None

    # Scaling scenarios
    scaling_inputs = {"Low Volume (100/day)": 100, "Medium Volume (10,000/day)": 10000, "High Volume (1M/day)": 1000000}
    scaling = {}
    for label, rpd in scaling_inputs.items():
        ph = compute_phased_costs(
            training_hours=training_hours,
            training_compute_cost_per_hour=pricing.compute_per_hour,
            dataset_size_gb=dataset_size_gb,
            training_data_transfer_gb=training_data_transfer_gb,
            inference_requests_per_day=rpd,
            avg_latency_seconds=avg_latency_seconds,
            compute_cost_per_hour_inference=pricing.compute_per_hour,
            model_size_gb=model_size_gb,
            deployment_months=deployment_months,
            pricing=pricing,
            avg_request_payload_gb=avg_request_payload_gb,
            avg_response_payload_gb=avg_response_payload_gb,
            days_per_month=days_per_month,
        )
        scaling[label] = ph["phase2_monthly"]["phase2_monthly_total"]

    # Build markdown according to the exact template
    def fmt_money(x):
        return f"${x:,.2f}"

    lines = []
    lines.append("===== RNN DEPLOYMENT COST ANALYSIS REPORT =====\n")
    lines.append(f"Project: {project_name}")
    lines.append(f"Cloud Provider: {provider}")
    lines.append(f"Region: {region}")
    lines.append(f"Analysis Date: {analysis_date}\n")

    lines.append("--- MODEL SPECIFICATIONS ---")
    lines.append(f"Architecture: {specs.get('architecture')}")
    lines.append(f"Parameters: {specs.get('parameters')}")
    lines.append(f"Model Size: {specs.get('model_size')}")
    lines.append(f"Dataset Size: {specs.get('dataset_size')}\n")

    lines.append("--- INFRASTRUCTURE SPECIFICATIONS ---")
    lines.append(f"Instance Type: {infra.get('instance_type')}")
    lines.append(f"vCPUs: {infra.get('vcpus')}")
    lines.append(f"RAM: {infra.get('ram')}")
    lines.append(f"Storage: {infra.get('storage')}\n")

    lines.append("--- COST BREAKDOWN ---\n")
    lines.append("1. TRAINING COSTS (One-time)")
    lines.append(f"   Compute: {fmt_money(phase1.get('compute_training',0))} ({phase1.get('compute_training',0)/ (pricing.compute_per_hour or 1):.2f} hours × ${pricing.compute_per_hour:.2f}/hour)")
    lines.append(f"   Storage: {fmt_money(phase1.get('storage_training',0))} ({dataset_size_gb} GB × ${pricing.storage_per_gb_month:.2f}/GB)")
    lines.append(f"   Data Transfer: {fmt_money(phase1.get('data_transfer_training',0))}")
    lines.append(f"   Total Training Cost: {fmt_money(phase1.get('phase1_total',0))}\n")

    lines.append("2. INFERENCE COSTS (Monthly)")
    lines.append(f"   Compute: {fmt_money(phase2.get('compute_monthly',0))} ({phase2.get('runtime_hours_month',0):.2f} hours × ${pricing.compute_per_hour:.2f}/hour)")
    lines.append(f"   Storage: {fmt_money(phase2.get('storage_monthly',0))}")
    lines.append(f"   Data Transfer: {fmt_money(phase2.get('data_transfer_monthly',0))} ({phase2.get('requests_per_month',0):,} requests × ${pricing.data_transfer_per_gb:.2f}/GB equivalent)")
    lines.append(f"   Total Monthly Cost: {fmt_money(phase2.get('phase2_monthly_total',0))}\n")

    lines.append("3. KEY METRICS")
    lines.append(f"   Cost per Inference: {fmt_money(cost_per_inference)}")
    lines.append(f"   Inferences per Dollar: {int(inferences_per_dollar) if inferences_per_dollar else 'N/A'}")
    lines.append(f"   Monthly Inference Capacity: {int(phase2.get('requests_per_month',0)):,} requests\n")

    lines.append("4. SCALING SCENARIOS")
    for label, monthly in scaling.items():
        lines.append(f"   {label}: {fmt_money(monthly)}/month")

    lines.append('\n--- ASSUMPTIONS ---')
    lines.append('• Assumed 99% uptime')
    lines.append('• Used on-demand pricing; reserved instances would reduce cost by ~40%')
    lines.append(f'• Assumed average inference time of {avg_latency_seconds*1000:.0f}ms based on observed metrics')
    lines.append(f'• Data transfer costs assume {(avg_request_payload_gb+avg_response_payload_gb)*1024:.1f} KB per request (request+response)')

    lines.append('\n--- OPTIMIZATION RECOMMENDATIONS ---')
    lines.append('• Consider spot instances for training to reduce costs by 70%')
    lines.append('• Implement model quantization to reduce inference cost')
    lines.append('• Use caching for repeated requests')

    lines.append('\n--- COST COMPARISON ---')
    lines.append('Alternative Model: Transformer (example)')
    lines.append('Cost Difference: +150% more expensive')
    lines.append('Performance Difference: +10% accuracy (example)')
    lines.append('Cost-Efficiency Trade-off: Transformers increase latency and cost; choose only if performance gain justifies spend')

    lines.append('\n==========================================')

    return "\n".join(lines)


def example_scenario() -> Dict:
    """Return an example cost breakdown for demonstration/testing."""
    pricing = Pricing(
        compute_per_hour=0.5,  # example cheap CPU instance
        storage_per_gb_month=0.02,
        data_transfer_per_gb=0.09,
        misc_monthly=10.0
    )

    return total_cost_of_ownership(
        training_hours=10.0,
        inference_requests_per_day=1000,
        avg_latency_seconds=0.2,
        model_size_gb=0.2,
        dataset_size_gb=5.0,
        deployment_months=6,
        pricing=pricing,
        avg_request_payload_gb=0.0001,
        avg_response_payload_gb=0.0005,
        additional_training_costs=5.0,
        other_one_time_costs=20.0
    )


if __name__ == "__main__":
    import json

    print("Example cost scenario:\n")
    es = example_scenario()
    print(json.dumps(es, indent=2))

    print("\nSensitivity for requests_per_day = [100, 1_000, 10_000]\n")
    pricing = Pricing(0.5, 0.02, 0.09, misc_monthly=10.0)
    sens = sensitivity_by_requests([100, 1000, 10000], avg_latency_seconds=0.2, pricing=pricing,
                                   other_params={"deployment_months": 6, "model_size_gb": 0.2, "dataset_size_gb": 5.0})
    print(json.dumps(sens, indent=2))

    print('\nPhased cost breakdown (Phase 1 = training; Phase 2 = monthly inference)\n')
    phased = compute_phased_costs(
        training_hours=10.0,
        training_compute_cost_per_hour=0.5,
        dataset_size_gb=5.0,
        training_data_transfer_gb=2.0,
        inference_requests_per_day=1000,
        avg_latency_seconds=0.2,
        compute_cost_per_hour_inference=0.5,
        model_size_gb=0.2,
        deployment_months=6,
        pricing=pricing,
        avg_request_payload_gb=0.0001,
        avg_response_payload_gb=0.0005,
        days_per_month=30,
    )

    print(json.dumps(phased, indent=2))
