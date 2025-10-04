"""
Quick diagnostic script to check Safebooru dataset structure
"""

import pandas as pd
from pathlib import Path
import kagglehub

print("Downloading Safebooru dataset...")
path = kagglehub.dataset_download("alamson/safebooru")
print(f"Path: {path}")

# Find CSV
csv_files = list(Path(path).glob("*.csv"))
if not csv_files:
    print("No CSV found!")
    exit(1)

csv_path = csv_files[0]
print(f"\nReading: {csv_path}")

# Load sample
df = pd.read_csv(csv_path, nrows=10)

print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print("="*100)

for idx, row in df.head(3).iterrows():
    print(f"\n--- Row {idx} ---")
    for col in df.columns:
        value = row[col]
        if pd.notna(value):
            # Truncate long strings
            str_val = str(value)
            if len(str_val) > 100:
                str_val = str_val[:97] + "..."
            print(f"  {col:20s}: {str_val}")

print("\n" + "="*100)
print("\n💡 Look for columns that might contain image URLs:")
print("   - file_url, sample_url, preview_url, url")
print("   - directory + image (to construct URL)")
print("   - md5 (might be used in URL construction)")
