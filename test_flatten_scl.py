"""Test that flattening produces diverse representations unlike mean pooling."""
import torch
from sci_arc.training.losses import StructuralContrastiveLoss

# Create SCL loss with flattening
scl = StructuralContrastiveLoss(
    hidden_dim=256, 
    projection_dim=128, 
    temperature=0.07,
    num_structure_slots=8
)
scl.train()

B = 32
K = 8
D = 256

print("=" * 60)
print("Test 1: Completely collapsed representations (pathological)")
print("=" * 60)
# This mimics extreme collapse: all samples have IDENTICAL slot values
base = torch.randn(1, K, D)
z_collapsed = base.expand(B, -1, -1).clone()  # Truly identical

transform_labels = torch.randint(0, 8, (B,))

z_test = z_collapsed.clone().requires_grad_(True)
loss = scl(z_test, transform_labels)
loss.backward()
print(f"Loss: {loss.item():.4f}")
print(f"Gradient norm: {z_test.grad.norm():.4f}")

print()
print("=" * 60)
print("Test 2: Realistic scenario - each sample has unique random features")
print("=" * 60)
# This is what we SHOULD see after 1 gradient update
z_diverse = torch.randn(B, K, D)  # Each sample is independently random

z_flat = z_diverse.reshape(B, -1)
z_flat_norm = torch.nn.functional.normalize(z_flat, dim=-1)
sim_flat = torch.mm(z_flat_norm, z_flat_norm.t())
mean_sim_flat = (sim_flat.sum() - B) / (B * B - B)
print(f"FLATTENED mean similarity: {mean_sim_flat:.4f}")

z_pooled = z_diverse.mean(dim=1)
z_pooled_norm = torch.nn.functional.normalize(z_pooled, dim=-1)
sim_pooled = torch.mm(z_pooled_norm, z_pooled_norm.t())
mean_sim_pooled = (sim_pooled.sum() - B) / (B * B - B)
print(f"POOLED mean similarity: {mean_sim_pooled:.4f}")

# Check SCL losses
losses = []
for _ in range(5):
    loss = scl(z_diverse, transform_labels)
    losses.append(loss.item())

print(f"SCL losses: {[f'{l:.4f}' for l in losses]}")
print(f"Mean: {sum(losses)/len(losses):.4f}, Std: {torch.tensor(losses).std():.4f}")

print()
print("=" * 60)
print("Test 3: Near-collapse but with small variations (realistic early training)")
print("=" * 60)
# Small but non-zero differences between samples
base = torch.randn(1, K, D)
z_near_collapse = base + torch.randn(B, K, D) * 0.1  # 10% variation

z_flat = z_near_collapse.reshape(B, -1)
z_flat_norm = torch.nn.functional.normalize(z_flat, dim=-1)
sim_flat = torch.mm(z_flat_norm, z_flat_norm.t())
mean_sim_flat = (sim_flat.sum() - B) / (B * B - B)
print(f"FLATTENED mean similarity: {mean_sim_flat:.4f}")

z_pooled = z_near_collapse.mean(dim=1)
z_pooled_norm = torch.nn.functional.normalize(z_pooled, dim=-1)
sim_pooled = torch.mm(z_pooled_norm, z_pooled_norm.t())
mean_sim_pooled = (sim_pooled.sum() - B) / (B * B - B)
print(f"POOLED mean similarity: {mean_sim_pooled:.4f}")

losses = []
for _ in range(5):
    loss = scl(z_near_collapse, transform_labels)
    losses.append(loss.item())
print(f"SCL losses: {[f'{l:.4f}' for l in losses]}")
print(f"Mean: {sum(losses)/len(losses):.4f}, Std: {torch.tensor(losses).std():.4f}")
