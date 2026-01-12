#!/usr/bin/env python3
"""Quick test to validate flow matching implementation for MNIST."""

import torch
from scaling_recipes.model import FlowMLP
from scaling_recipes.loss import ConditionalFlowMatchingLoss
from scaling_recipes.datasets import MNISTDataset

def test_flow_matching():
    print("Testing Flow Matching Implementation...")

    # Test 1: Model instantiation
    print("\n1. Testing model instantiation...")
    model = FlowMLP(n_features=784, width=32, n_blocks=5, parametrization="mup")
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test 2: Forward pass with random data
    print("\n2. Testing forward pass with random data...")
    batch_size = 10
    x = torch.randn(batch_size, 784)
    t = torch.rand(batch_size)
    output = model(x, t)
    assert output.shape == (batch_size, 784), f"Expected shape {(batch_size, 784)}, got {output.shape}"
    print(f"   ✓ Forward pass successful, output shape: {output.shape}")

    # Test 3: Loss computation
    print("\n3. Testing loss computation...")
    loss_fn = ConditionalFlowMatchingLoss(sigma_min=1e-4)
    loss = loss_fn(model, x)
    assert not torch.isnan(loss), "Loss is NaN!"
    print(f"   ✓ Loss computation successful: {loss.item():.6f}")

    # Test 4: Backward pass
    print("\n4. Testing backward pass...")
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed!"
    print(f"   ✓ Backward pass successful, gradients computed")

    # Test 5: MNIST dataset loading
    print("\n5. Testing MNIST dataset loading...")
    dataset = MNISTDataset(size=1000, batch_size=100)
    train_loader = dataset.create(type="train")
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape[1] == 784, f"Expected 784 features, got {x_batch.shape[1]}"
    print(f"   ✓ Dataset loaded, batch shape: {x_batch.shape}")

    # Test 6: Loss with real data
    print("\n6. Testing loss with real MNIST data...")
    model.eval()
    with torch.no_grad():
        loss = loss_fn(model, x_batch)
    print(f"   ✓ Loss on real data: {loss.item():.6f}")

    # Test 7: Different widths and parametrizations
    print("\n7. Testing different configurations...")
    for param in ["mup", "sp"]:
        for width in [16, 32, 64]:
            model_test = FlowMLP(n_features=784, width=width, n_blocks=3, parametrization=param)
            x_test = torch.randn(5, 784)
            out_test = model_test(x_test)
            assert out_test.shape == (5, 784)
    print(f"   ✓ All configurations working")

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)

if __name__ == "__main__":
    test_flow_matching()
