# Flow Matching for MNIST - Implementation Guide

## Overview

This implementation adds flow matching capabilities for MNIST generative modeling with support for scaling experiments across different model widths. The implementation follows the same parametrization approach (muP and SP) as the classification task.

## What Was Implemented

### 1. Model Architecture (`scaling_recipes/model.py`)

**FlowMLP** - Adapted for MNIST:
- **Input**: 784 features (flattened 28x28 MNIST images)
- **Architecture**:
  - Time embedding: Uses 1/4 of width for sinusoidal time encoding
  - First block: Maps from (n_features + time_embedding_size) to width
  - Residual blocks: Preserve width through residual connections
  - Final layer: Maps width back to 784 features
- **Parametrizations**:
  - **muP**: Scaled initialization and per-layer learning rates
  - **SP**: Standard Xavier initialization

### 2. Loss Function (`scaling_recipes/loss.py`)

**ConditionalFlowMatchingLoss**:
- Implements conditional flow matching for generative modeling
- Uses linear interpolation between noise and data
- Compatible with MNIST dataloader format (x, y) tuples

### 3. Training & Evaluation (`scaling_recipes/main.py`)

**New Functions**:
- `compute_flow_validation_loss()`: Computes validation loss for flow models
- `train_flow()`: Training function with learning rate scheduling
- `evaluate_flow()`: Evaluation on test set
- `sweep_train_flow()`: Sweep across widths and learning rates
- `sweep_evaluate_flow()`: Evaluate sweep results and generate plots
- `sweep_train_and_evaluate_flow()`: Complete pipeline

### 4. Visualization (`scaling_recipes/util.py`)

**flow_sweep_plot()**:
- Plots test loss vs learning rate
- Color-coded by model width using viridis colormap
- Log-log scale for better visualization
- Power-of-2 colorbar for width values
- Smoothing applied for cleaner curves

### 5. Configuration (`scaling_recipes/cli/conf/base.yaml`)

**New Sections**:
- `flow_model`: FlowMLP configuration
- `flow_trainer`: Training settings for flow matching
- `flow_logger`: Logging configuration
- `flow_sweep`: Sweep parameters (widths: [8, 16, 32, 64, 128])

### 6. CLI Commands (`pyproject.toml`)

**New Commands**:
- `train_flow`: Train a single flow matching model
- `evaluate_flow`: Evaluate a trained flow model
- `sweep_flow`: Run complete sweep (train + evaluate)
- `sweep_evaluate_flow`: Evaluate existing sweep checkpoints

## Usage

### Single Model Training

```bash
python -m scaling_recipes.main train_flow flow_model.width=64 flow_trainer.optimizer.lr=0.001
```

### Sweep Training

```bash
python -m scaling_recipes.main sweep_train_and_evaluate_flow
```

This will:
1. Train models at different widths: [8, 16, 32, 64, 128]
2. Test learning rates: 10 values from 2^-10 to 2^-1
3. Save checkpoints for each configuration
4. Generate final plot: `sweep_plots/flow_mup_sweep.png`

### Evaluate Existing Sweep

```bash
python -m scaling_recipes.main sweep_evaluate_flow
```

## Expected Output

The final plot (`sweep_plots/flow_mup_sweep.png`) will show:
- **Y-axis**: Test loss (log scale)
- **X-axis**: Learning rate (log scale)
- **Colors**: Different widths (viridis colormap)
- **Colorbar**: Powers of 2 (2^3, 2^4, 2^5, etc.)

## Implementation Details

### Model Dimensions

For a model with width W:
- Time embedding dimension: W // 4
- First layer: (784 + W//4) → W
- Hidden layers: W → W (with residual connections)
- Final layer: W → 784

### Learning Rate Scaling (muP)

- Hidden layers: `lr * (32 / width)`
- Time embedding: `lr * 0.3`
- Weight decay: `wd * (width / 1024)`

### Training Configuration

- **Epochs**: 25
- **Batch size**: 10,000
- **Dataset size**: 50,000 training samples
- **Optimizer**: AdamW with adaptive betas
- **LR scheduler**: Linear decay
- **Gradient clipping**: 1.0

## File Structure

```
scaling_recipes/
├── model.py                    # FlowMLP model
├── loss.py                     # ConditionalFlowMatchingLoss
├── main.py                     # Training/evaluation functions
├── util.py                     # flow_sweep_plot()
└── cli/conf/base.yaml          # Configuration

logs/
└── mnist_flow/                 # Flow matching logs
    ├── mup_<width>_lr<lr>/
    │   ├── loss/
    │   └── ckpt/

sweep_plots/
└── flow_mup_sweep.png          # Final sweep plot
```

## Comparison with Classification

| Aspect | Classification | Flow Matching |
|--------|---------------|---------------|
| Model | MLP | FlowMLP |
| Loss | CrossEntropyLoss | ConditionalFlowMatchingLoss |
| Output | 10 classes | 784 features |
| Metrics | Loss + Accuracy | Loss only |
| Task | Discriminative | Generative |

## Next Steps

1. Install dependencies: `pip install -e .`
2. Run sweep: `python -m scaling_recipes.main sweep_train_and_evaluate_flow`
3. Check results: `sweep_plots/flow_mup_sweep.png`
4. Compare muP vs SP: Change `flow_model.parametrization: sp` in config

## Technical Notes

- **Time embedding**: Uses sinusoidal encoding with frequencies π * [1, 2, ..., dim//2]
- **Residual connections**: Help with gradient flow in deep networks
- **Smoothing**: Rolling window (size=5) applied to loss curves for better visualization
- **Memory management**: Models are deleted and CUDA cache cleared after each training run
