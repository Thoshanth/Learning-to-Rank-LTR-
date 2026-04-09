"""
RUN ALL STEPS IN ORDER
=======================
Run this file to execute the full Project 1 pipeline.
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

steps = [
    ("Step 1: Generate Dataset",       "data/generate_data.py"),
    ("Step 2: EDA + BM25 Baseline",    "step2_eda_baseline.py"),
    ("Step 3: Train Model",            "step3_train_model.py"),
    ("Step 4: Evaluate + Resume",      "step4_evaluate_resume.py"),
]

print("=" * 55)
print("PROJECT 1: SEARCH RANKING MODEL — FULL PIPELINE")
print("=" * 55)

for step_name, script in steps:
    print(f"\n{'='*55}")
    print(f">>> {step_name}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR in {script}. Fix before continuing.")
        sys.exit(1)

print("\n" + "="*55)
print("ALL STEPS COMPLETE!")
print("Check the outputs/ folder for all plots.")
print("="*55)
