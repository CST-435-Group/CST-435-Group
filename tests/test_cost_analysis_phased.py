import json
from cost_analysis import compute_phased_costs, Pricing


def test_phased_basic():
    pricing = Pricing(compute_per_hour=0.5, storage_per_gb_month=0.02, data_transfer_per_gb=0.09, misc_monthly=10.0)
    res = compute_phased_costs(
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

    assert "phase1" in res
    assert "phase2_monthly" in res
    assert res["phase1"]["compute_training"] == 5.0  # 10h * $0.5
    # runtime hours per month = 1000 * 30 * 0.2 / 3600 = approx 1.6667 hours
    assert res["phase2_monthly"]["runtime_hours_month"] > 1.6
    assert res["phase2_monthly"]["phase2_monthly_total"] > 0


if __name__ == '__main__':
    # quick local run
    test_phased_basic()
    print('test passed')
