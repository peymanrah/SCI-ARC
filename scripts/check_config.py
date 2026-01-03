#!/usr/bin/env python3
"""Quick YAML validation script."""
import yaml
import sys

config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/rlan_stable_dev.yaml'
config = yaml.safe_load(open(config_file))
print(f'YAML valid: {config_file}')
print(f"use_merged_training: {config['data'].get('use_merged_training', False)}")
print(f"num_cached_samples: {config['data'].get('num_cached_samples', 'not set')}")
print(f"samples_per_task: {config['data'].get('samples_per_task', 50)}")
print(f"checkpoint_dir: {config['logging'].get('checkpoint_dir', 'not set')}")
print(f"hpm_buffer_path: {config['model'].get('hpm_buffer_path', 'not set')}")
