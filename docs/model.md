# VEYRA-LM: Neural Architecture

VEYRA-LM is a modern decoder-only autoregressive language model implemented entirely from deep learning primitives in PyTorch.

## Key Design Principles

1. **RMSNorm (Root Mean Square Layer Normalization)**:
   - Eliminates mean-centering overhead from standard LayerNorm.
   - Provides smooth, scale-invariant forward and backward dynamics.
2. **RoPE (Rotary Positional Embeddings)**:
   - Rotates query and key representations in pairwise 2D complex subspaces.
   - Avoids rigid learned absolute positional lookup tables, providing natural length generalization.
3. **Causal Multi-Head & Grouped-Query Attention (GQA)**:
   - Strictly applies lower-triangular causal masks so past tokens never attend to future tokens.
   - Supports Grouped-Query Attention (`num_kv_heads < num_heads`) for low-memory KV caching.
4. **SwiGLU Feed-Forward Network**:
   - Computes `w2(SiLU(w1(x)) * w3(x))`.
   - Offers superior expressivity and training speed over traditional GELU/ReLU layers.
5. **Stabilized Weight Initialization**:
   - Embeddings and linear weights initialized from $\mathcal{N}(0, 0.02)$.
   - Residual projections scaled by $1 / \sqrt{2 \times \text{num\_layers}}$ to preserve gradient magnitude across deep layers.
