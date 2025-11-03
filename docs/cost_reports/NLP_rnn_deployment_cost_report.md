===== RNN DEPLOYMENT COST ANALYSIS REPORT =====

Project: NLP
Cloud Provider: [AWS/Azure/GCP/Other]
Region: [e.g., US-East]
Analysis Date: 2025-11-02

--- MODEL SPECIFICATIONS ---
Architecture: (unknown - please fill)
Parameters: (unknown - please fill)
Model Size: 0.00 B
Dataset Size: 261.34 KB

Dataset sources:
- NLP\data — 261.34 KB

--- INFRASTRUCTURE SPECIFICATIONS ---
Instance Type: [e.g., c5.2xlarge]
vCPUs: [e.g., 8 cores]
RAM: [e.g., 16 GB]
Storage: [e.g., 50 GB SSD]

--- COST BREAKDOWN ---

1. TRAINING COSTS (One-time)
   Compute: $10.52 (20 hours × $0.526/hour)
   Storage: $0.00 (0.00 GB × $0.08/GB-month)
   Data Transfer: $0.00
   Total Training Cost: $10.52

2. INFERENCE COSTS (Monthly)
   Compute: $10.52 (20 hours × $0.526/hour)
   Storage: $0.00
   Data Transfer: $0.00 ([X] GB × $Y.YY/GB)
   Total Monthly Cost: $30.21 (example for 10k/day)

3. KEY METRICS
   Cost per Inference: $0.000000
   Inferences per Dollar: 2,423,076
   Monthly Inference Capacity: 72,576,000 requests

4. SCALING SCENARIOS
   Low Volume (100/day): $29.95/month
   Medium Volume (10,000/day): $30.21/month
   High Volume (1M/day): $55.70/month

--- ASSUMPTIONS ---
• [List all assumptions made, e.g., "Assumed 99% uptime"]
• [e.g., "Used on-demand pricing; reserved instances would reduce cost by ~40%"]
• [e.g., "Assumed average inference time of 50ms based on local testing"]
• [e.g., "Data transfer costs assume 10 KB per request"]

--- OPTIMIZATION RECOMMENDATIONS ---
• [e.g., "Consider spot instances for training to reduce costs by 70%"]
• [e.g., "Implement model quantization to reduce inference cost"]
• [e.g., "Use caching for repeated requests"]

--- COST COMPARISON ---
[Compare with alternative approaches if applicable]
Alternative Model: [e.g., Transformer model]
Cost Difference: [e.g., "+150% more expensive"]
Performance Difference: [e.g., "+10% accuracy"]
Cost-Efficiency Trade-off: [Analysis]

==========================================

Detected config files (check for parameters/architecture):
- NLP\nlp-react\saved_model\tokenizer_config.json
- NLP\model\mlm_hospital_reviews\final_model\tokenizer_config.json

--- Computation assumptions used to fill defaults ---
• Provider default: AWS us-east-1
• Training instance: g4dn.xlarge at $0.526/hour
• Inference instance: t3.medium at $0.0416/hour
• Storage rate: $0.08/GB-month
• Data transfer rate: $0.09/GB
• Assumed training time: 20 hours
• Assumed average inference time: 50 ms
• Assumed data per request: 10 KB
