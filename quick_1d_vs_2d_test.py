"""
Quick Test: 1D vs 2D Structure Extraction
Fast version for immediate results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("QUICK TEST: 1D Rasterization vs 2D Grid-Based")
print("=" * 60)

# ============ DATA ============
def make_data(n=100):
    """Generate simple test data."""
    inputs, outputs, labels = [], [], []
    for i in range(n):
        grid = np.zeros((5, 5), dtype=np.int64)
        grid[1:4, 2] = 1  # vertical line
        
        label = i % 4
        if label == 0:  # rotate 90
            out = np.rot90(grid, 1)
        elif label == 1:  # flip h
            out = np.fliplr(grid)
        elif label == 2:  # flip v
            out = np.flipud(grid)
        else:  # identity
            out = grid.copy()
        
        inputs.append(grid)
        outputs.append(out)
        labels.append(label)
    
    return (
        torch.tensor(np.stack(inputs)),
        torch.tensor(np.stack(outputs)),
        torch.tensor(labels)
    )

inputs, outputs, labels = make_data(100)
print(f"Data: {len(inputs)} samples, 4 transforms")

# ============ MODELS ============

class Model1D(nn.Module):
    """1D Rasterization approach."""
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.Embedding(10, d)
        self.pos = nn.Embedding(50, d)  # 25 + 25 for in/out
        self.tf = nn.TransformerEncoderLayer(d, 2, d*2, batch_first=True)
        self.out = nn.Linear(d, d)
    
    def forward(self, inp, out):
        B = inp.shape[0]
        x = torch.cat([inp.flatten(1), out.flatten(1)], dim=1)  # [B, 50]
        pos_idx = torch.arange(50, device=x.device).unsqueeze(0)
        e = self.emb(x) + self.pos(pos_idx)  # [B, 50, D]
        h = self.tf(e)
        return self.out(h.mean(dim=1))  # [B, D]

class Model2D(nn.Module):
    """2D Grid approach."""
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.Embedding(10, d)
        self.pos_x = nn.Embedding(10, d//2)
        self.pos_y = nn.Embedding(10, d//2)
        self.pos_proj = nn.Linear(d, d)
        self.diff_proj = nn.Linear(d*2, d)
        self.tf = nn.TransformerEncoderLayer(d, 2, d*2, batch_first=True)
        self.out = nn.Linear(d, d)
    
    def forward(self, inp, out):
        B, H, W = inp.shape
        
        # Embed
        inp_e = self.emb(inp)  # [B, H, W, D]
        out_e = self.emb(out)
        
        # 2D pos
        y_idx = torch.arange(H, device=inp.device)
        x_idx = torch.arange(W, device=inp.device)
        y_emb = self.pos_y(y_idx).unsqueeze(1).expand(-1, W, -1)
        x_emb = self.pos_x(x_idx).unsqueeze(0).expand(H, -1, -1)
        pos = self.pos_proj(torch.cat([y_emb, x_emb], dim=-1))  # [H, W, D]
        
        inp_e = inp_e + pos
        out_e = out_e + pos
        
        # Diff
        diff = self.diff_proj(torch.cat([
            inp_e.flatten(1, 2), out_e.flatten(1, 2)
        ], dim=-1))  # [B, H*W, D]
        
        h = self.tf(diff)
        return self.out(h.mean(dim=1))  # [B, D]

# ============ TRAIN ============

def train(model, name, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for ep in range(epochs):
        model.train()
        z = model(inputs, outputs)  # [B, D]
        z = F.normalize(z, dim=-1)
        
        # Contrastive
        sim = z @ z.t() / 0.1
        
        # Positive mask
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag = torch.eye(len(labels)).bool()
        pos_mask = labels_eq & ~diag
        neg_mask = ~labels_eq
        
        # Loss
        exp_sim = torch.exp(sim)
        exp_sim_masked = exp_sim.masked_fill(diag, 0)
        pos_sum = (exp_sim * pos_mask.float()).sum(1)
        all_sum = exp_sim_masked.sum(1)
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8)).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # Eval
    model.eval()
    with torch.no_grad():
        z = model(inputs, outputs)
        z = F.normalize(z, dim=-1)
        sim = z @ z.t()
        
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag = torch.eye(len(labels)).bool()
        pos_mask = labels_eq & ~diag
        neg_mask = ~labels_eq
        
        same_sim = sim[pos_mask].mean().item()
        diff_sim = sim[neg_mask].mean().item()
        sep = same_sim - diff_sim
    
    print(f"\n{name}:")
    print(f"  Same-class similarity: {same_sim:.4f}")
    print(f"  Diff-class similarity: {diff_sim:.4f}")
    print(f"  SEPARATION: {sep:.4f}")
    return sep

sep_1d = train(Model1D(), "1D Rasterization")
sep_2d = train(Model2D(), "2D Grid-Based")

print("\n" + "=" * 60)
print("RESULT:")
if sep_1d > sep_2d:
    print(f"  1D WINS by {sep_1d - sep_2d:.4f}")
else:
    print(f"  2D WINS by {sep_2d - sep_1d:.4f}")
print("=" * 60)

print("""
INTERPRETATION:
- 'Separation' = how well model distinguishes transforms
- Higher = better clustering of same-transform tasks
- Both can learn, but 2D typically has edge for spatial transforms
""")
