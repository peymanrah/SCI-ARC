#!/usr/bin/env python3
"""Quick validation of rlan_stable_prod.yaml"""

import yaml
import sys

print("=" * 60)
print("RLAN_STABLE_PROD.YAML VALIDATION")
print("=" * 60)

try:
    with open('configs/rlan_stable_prod.yaml') as f:
        c = yaml.safe_load(f)
    print("YAML syntax: OK")
except Exception as e:
    print(f"YAML ERROR: {e}")
    sys.exit(1)

t = c['training']
m = c['model']

print("\n--- Critical DSC Fix Parameters ---")
lcd = t.get('lambda_centroid_diversity')
print(f"lambda_centroid_diversity: {lcd}")
if lcd and lcd >= 0.5:
    print("  STATUS: OK - properly set to prevent DSC collapse")
elif lcd:
    print(f"  WARNING: {lcd} may be too low, recommend 0.5+")
else:
    print("  CRITICAL: MISSING! This will cause DSC collapse!")
    sys.exit(1)

print("\n--- Model Config ---")
print(f"use_dsc: {m.get('use_dsc')}")
print(f"use_msre: {m.get('use_msre')}")
print(f"max_clues: {m.get('max_clues')}")
print(f"num_solver_steps: {m.get('num_solver_steps')}")
print(f"hidden_dim: {m.get('hidden_dim')}")
print(f"use_hyperlora: {m.get('use_hyperlora')}")

print("\n--- Training Config ---")
print(f"batch_size: {t.get('batch_size')}")
print(f"gradient_accumulation_steps: {t.get('gradient_accumulation_steps')}")
print(f"learning_rate: {t.get('learning_rate')}")
print(f"max_epochs: {t.get('max_epochs')}")
print(f"loss_mode: {t.get('loss_mode')}")

print("\n--- Loss Lambdas ---")
print(f"lambda_entropy: {t.get('lambda_entropy')}")
print(f"lambda_sparsity: {t.get('lambda_sparsity')}")
print(f"lambda_predicate: {t.get('lambda_predicate')}")
print(f"lambda_centroid_diversity: {t.get('lambda_centroid_diversity')}")

print("\n" + "=" * 60)
print("CONFIG VALIDATED - Ready for training!")
print("=" * 60)
