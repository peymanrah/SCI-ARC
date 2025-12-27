"""
Quick smoke test for RLAN model.
"""
import torch


def test_rlan_forward_training_smoke():
    """Smoke test that RLAN forward_training works on CPU."""
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(hidden_dim=64)
    model = RLAN(config=config)
    
    # Create sample inputs
    batch_size = 2
    num_pairs = 3
    H, W = 10, 10
    
    x = torch.randint(0, 10, (batch_size, num_pairs, H, W))
    y = torch.randint(0, 10, (batch_size, num_pairs, H, W))
    t = torch.randint(0, 10, (batch_size, H, W))
    
    # Forward pass
    model.train()
    out = model.forward_training(x, y, t)
    
    # Verify output structure
    assert isinstance(out, dict), "forward_training should return dict"
    assert "logits" in out, "Output should contain 'logits'"
    assert out["logits"].shape == (batch_size, 10, H, W), f"Unexpected logits shape: {out['logits'].shape}"
    
    # Verify gradient flow
    loss = out["logits"].sum()
    loss.backward()
    
    # At least some parameters should have gradients
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "At least some parameters should have gradients"
    
    print("RLAN forward_training smoke test passed!")


def test_rlan_predict_smoke():
    """Smoke test that RLAN predict works on CPU."""
    from sci_arc.models.rlan import RLAN, RLANConfig
    
    config = RLANConfig(hidden_dim=64)
    model = RLAN(config=config)
    
    # Create sample inputs
    batch_size = 2
    num_pairs = 3
    H, W = 10, 10
    
    train_inputs = torch.randint(0, 10, (batch_size, num_pairs, H, W))
    train_outputs = torch.randint(0, 10, (batch_size, num_pairs, H, W))
    test_input = torch.randint(0, 10, (batch_size, H, W))
    
    # Predict
    model.eval()
    with torch.no_grad():
        preds = model.predict(test_input, train_inputs=train_inputs, train_outputs=train_outputs)
    
    assert preds.shape == (batch_size, H, W), f"Unexpected predictions shape: {preds.shape}"
    assert preds.dtype == torch.long, "Predictions should be long integers"
    assert (preds >= 0).all() and (preds < 10).all(), "Predictions should be in [0, 9]"
    
    print("RLAN predict smoke test passed!")


if __name__ == "__main__":
    test_rlan_forward_training_smoke()
    test_rlan_predict_smoke()
