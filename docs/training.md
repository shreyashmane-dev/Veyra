# VEYRA-LM Training Pipeline

The training pipeline supports end-to-end learning from randomly initialized weights on CPU or CUDA GPU.

## Key Features

1. **Cosine Learning Rate Schedule with Warmup**:
   - Linear warmup from `min_lr` to `base_lr`.
   - Cosine decay towards `min_lr`.
2. **Decoupled Weight Decay**:
   - Applies weight decay strictly to 2D weight matrices (embeddings and projection weights).
   - Exempts 1D normalization gain parameters to avoid numerical degradation.
3. **Gradient Accumulation & Clipping**:
   - Emulates large effective batch sizes on constrained consumer GPUs or CPUs.
   - Restricts gradient norm via `clip_grad_norm_` to protect against exploding gradients.
4. **Training Safeguards**:
   - Detects NaNs and Infs in loss computation and diagnoses errors immediately.
   - Periodic atomic checkpoint saving (`latest.pt` and `best.pt`).

## Running Training

```bash
python -m veyra.cli.main train --config configs/training/pretrain_tiny.yaml
```
