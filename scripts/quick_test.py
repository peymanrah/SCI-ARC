"""Quick test to verify RLAN model attribute access."""
import sys
sys.path.insert(0, 'c:/Users/perahmat/Downloads/SCI-ARC')

from sci_arc.models.rlan import RLAN, RLANConfig
import torch
from pathlib import Path

print("Loading checkpoint...")
checkpoint_path = Path('c:/Users/perahmat/Downloads/SCI-ARC/checkpoints/warmup3.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model_config = checkpoint.get('config', {}).get('model', {})

print(f"Full model config: {model_config}")

# Copy all model config params from checkpoint
print(f"dsc_use_complexity_signals in config: {'dsc_use_complexity_signals' in model_config}")
# Add the missing key - it was False during training
if 'dsc_use_complexity_signals' not in model_config:
    model_config['dsc_use_complexity_signals'] = False
config = RLANConfig(**{k: v for k, v in model_config.items() 
                       if k in RLANConfig.__dataclass_fields__ and k != 'type'})
print(f"RLANConfig: {config}")

print("Creating model...")
model = RLAN(config=config)
missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

# Check attribute access
print(f"\nModel attributes:")
print(f"  model.max_clues: {model.max_clues}")
print(f"  model.num_classes: {model.num_classes}")
print(f"  model.hidden_dim: {model.hidden_dim}")
print(f"  Has hyper_lora attr: {hasattr(model, 'hyper_lora')}")
if hasattr(model, 'hyper_lora'):
    print(f"  hyper_lora is None: {model.hyper_lora is None}")
print(f"  Has dsc: {model.dsc is not None}")
print(f"  Has msre: {model.msre is not None}")

print("\nDone!")
