===== RNN DEPLOYMENT COST ANALYSIS REPORT =====

Project: rnn-text-generator
Cloud Provider: [AWS/Azure/GCP/Other]
Region: [e.g., US-East]
Analysis Date: 2025-11-02

--- MODEL SPECIFICATIONS ---
Architecture: LSTM
Parameters: (unknown - please fill)
Model Size: 83.76 MB

Model files:
- rnn-text-generator\backend\saved_models\model_best.pt — 18.21 MB
- rnn-text-generator\backend\saved_models\model_val_acc_20.pt — 10.86 MB
- rnn-text-generator\backend\saved_models\model_val_acc_20_20251027_204408.pt — 18.21 MB
- rnn-text-generator\backend\saved_models\model_val_acc_21.pt — 18.21 MB
- rnn-text-generator\backend\saved_models\model_val_acc_21_20251027_173507.pt — 18.21 MB
- rnn-text-generator\frontend\node_modules\.bin — 48.00 KB
- rnn-text-generator\frontend\node_modules\@babel\core\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@babel\eslint-parser\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@babel\helper-compilation-targets\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@babel\helper-create-class-features-plugin\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@babel\helper-create-regexp-features-plugin\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@babel\plugin-transform-runtime\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@babel\preset-env\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@eslint\eslintrc\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\@surma\rollup-plugin-off-main-thread\tests\fixtures\assets-in-worker\build\assets\my-asset-620b911b.bin — 12.00 B
- rnn-text-generator\frontend\node_modules\acorn-globals\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\babel-plugin-polyfill-corejs2\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\eslint\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\eslint-plugin-import\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\eslint-plugin-react\node_modules\.bin — 4.00 KB
- rnn-text-generator\frontend\node_modules\global-prefix\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\istanbul-lib-instrument\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\jsonpath\node_modules\.bin — 4.00 KB
- rnn-text-generator\frontend\node_modules\make-dir\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\postcss-load-config\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\postcss-svgo\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\react-scripts\node_modules\.bin — 4.00 KB
- rnn-text-generator\frontend\node_modules\static-eval\node_modules\.bin — 4.00 KB
- rnn-text-generator\frontend\node_modules\sucrase\node_modules\.bin — 0.00 B
- rnn-text-generator\frontend\node_modules\tsconfig-paths\node_modules\.bin — 0.00 B
Dataset Size: 124.29 MB

Dataset sources:
- rnn-text-generator\backend\data — 124.29 MB

--- INFRASTRUCTURE SPECIFICATIONS ---
Instance Type: [e.g., c5.2xlarge]
vCPUs: [e.g., 8 cores]
RAM: [e.g., 16 GB]
Storage: [e.g., 50 GB SSD]

--- COST BREAKDOWN ---

1. TRAINING COSTS (One-time)
   Compute: $10.52 (20 hours × $0.526/hour)
   Storage: $0.02 (0.20 GB × $0.08/GB-month)
   Data Transfer: $0.01
   Total Training Cost: $10.55

2. INFERENCE COSTS (Monthly)
   Compute: $10.52 (20 hours × $0.526/hour)
   Storage: $0.01
   Data Transfer: $0.01 ([X] GB × $Y.YY/GB)
   Total Monthly Cost: $30.22 (example for 10k/day)

3. KEY METRICS
   Cost per Inference: $0.000000
   Inferences per Dollar: 2,423,076
   Monthly Inference Capacity: 72,576,000 requests

4. SCALING SCENARIOS
   Low Volume (100/day): $29.96/month
   Medium Volume (10,000/day): $30.22/month
   High Volume (1M/day): $55.71/month

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
- rnn-text-generator\backend\saved_models\model_val_acc_20_20251027_204408_config.json
- rnn-text-generator\backend\saved_models\model_val_acc_20_config.json
- rnn-text-generator\backend\saved_models\model_val_acc_21_20251027_173507_config.json
- rnn-text-generator\backend\saved_models\model_val_acc_21_config.json

--- Computation assumptions used to fill defaults ---
• Provider default: AWS us-east-1
• Training instance: g4dn.xlarge at $0.526/hour
• Inference instance: t3.medium at $0.0416/hour
• Storage rate: $0.08/GB-month
• Data transfer rate: $0.09/GB
• Assumed training time: 20 hours
• Assumed average inference time: 50 ms
• Assumed data per request: 10 KB
