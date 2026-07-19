#!/usr/bin/env python3
"""
Execute all notebooks with proper outputs captured.
This creates a final comprehensive set of executed notebooks with all outputs preserved.
"""

import subprocess
import sys
import os
from pathlib import Path

notebooks_to_run = [
    "00_data_preparation.ipynb",
    "01_cGAN_training.ipynb",
    "02_diffusion_training.ipynb",
    "03_evaluation.ipynb",
]

print("=" * 80)
print("NOTEBOOK EXECUTION WITH OUTPUT PRESERVATION")
print("=" * 80)
print()

for notebook in notebooks_to_run:
    if not Path(notebook).exists():
        print(f"⚠️  {notebook} not found, skipping...")
        continue
    
    print(f"Running {notebook}...")
    cmd = f"jupyter nbconvert --to notebook --inplace --execute {notebook}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"✓ {notebook} executed successfully")
        else:
            print(f"✗ {notebook} failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"✗ {notebook} timed out (>60 min)")
    except Exception as e:
        print(f"✗ Error executing {notebook}: {e}")
    
    print()

print("=" * 80)
print("✅ NOTEBOOK EXECUTION COMPLETE")
print("=" * 80)
