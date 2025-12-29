
import torch
import torch.nn as nn
import unittest
from sci_arc.models.rlan_modules.recursive_solver import RecursiveSolver

class TestSmartCheckpointing(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B, self.D, self.H, self.W = 2, 32, 10, 10
        self.hidden_dim = 32
        
        # Initialize solver with checkpointing enabled
        self.solver = RecursiveSolver(
            hidden_dim=self.hidden_dim,
            num_classes=10,
            # max_grid_size=30,  <-- REMOVED: Not an argument for RecursiveSolver
            gradient_checkpointing=True
        ).to(self.device)
        
        # Dummy inputs
        # clue_features shape: (B, K, D, H, W) where K is num_clues
        self.clue_features = torch.randn(self.B, 7, self.hidden_dim, self.H, self.W).to(self.device)
        self.count_embedding = torch.randn(self.B, self.hidden_dim).to(self.device)
        self.predicates = torch.randn(self.B, 8).to(self.device)
        self.input_grid = torch.randint(0, 10, (self.B, self.H, self.W)).to(self.device)
        
        # Dummy LoRA deltas (simulating HyperLoRA output)
        # These must require grad to verify backprop
        # Keys must match what ConvGRUCell expects: gru_reset, gru_update, gru_candidate
        # Shape must be (B, D, D) for _apply_lora_spatial
        self.lora_deltas = {
            'gru_reset': torch.randn(self.B, self.hidden_dim, self.hidden_dim, requires_grad=True, device=self.device),
            'gru_update': torch.randn(self.B, self.hidden_dim, self.hidden_dim, requires_grad=True, device=self.device),
            'gru_candidate': torch.randn(self.B, self.hidden_dim, self.hidden_dim, requires_grad=True, device=self.device)
        }

    def test_checkpointing_with_lora(self):
        """Verify that gradients flow to LoRA weights through the checkpoint."""
        print("\n[Test] Smart Checkpointing WITH LoRA...")
        
        self.solver.train() # Checkpointing only active in train mode
        
        # Forward pass
        outputs = self.solver(
            clue_features=self.clue_features,
            count_embedding=self.count_embedding,
            predicates=self.predicates,
            input_grid=self.input_grid,
            lora_deltas=self.lora_deltas
        )
        
        # Compute loss
        loss = outputs.mean()
        loss.backward()
        
        # Check gradients
        print("  Checking gradients on LoRA deltas...")
        for key, delta in self.lora_deltas.items():
            grad_norm = delta.grad.norm().item() if delta.grad is not None else 0.0
            print(f"    {key}: grad_norm={grad_norm:.6f}")
            self.assertIsNotNone(delta.grad, f"Gradient missing for {key}")
            self.assertGreater(grad_norm, 0.0, f"Gradient zero for {key}")
            
        print("  ✅ Success: Gradients flowed through checkpoint!")

    def test_checkpointing_without_lora(self):
        """Verify backward compatibility (no LoRA)."""
        print("\n[Test] Smart Checkpointing WITHOUT LoRA...")
        
        self.solver.train()
        
        # Forward pass
        outputs = self.solver(
            clue_features=self.clue_features,
            count_embedding=self.count_embedding,
            predicates=self.predicates,
            input_grid=self.input_grid,
            lora_deltas=None
        )
        
        loss = outputs.mean()
        loss.backward()
        print("  ✅ Success: Forward/Backward ran without error")

if __name__ == '__main__':
    unittest.main()
