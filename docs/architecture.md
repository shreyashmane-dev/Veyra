# VEYRA System Architecture

VEYRA is a personal artificial intelligence system designed around a clean separation of concerns:

```
USER
 ↓
TERMINAL / CLI (veyra)
 ↓
INPUT PROCESSING & PARSING
 ↓
CONTEXT ENGINE (Working Memory + Recent Conversation)
 ↓
VEYRA-LM (Self-Trained Transformer Engine)
 ↓
COGNITIVE ORCHESTRATOR
 ↓
LONG-TERM MEMORY (SQLite) / TOOLS / PERMISSION SECURITY
 ↓
RESPONSE GENERATION
```

## Architectural Decoupling

1. **VEYRA-LM**: A from-scratch autoregressive Transformer decoder model with RoPE, RMSNorm, and SwiGLU. Trainable on consumer hardware and scalable to 125M, 350M, and 1B parameters.
2. **Cognitive Core**: Orchestrates memory recall, user teaching, identity, and delegates unstructured generation to the active language engine.
3. **Memory Layer**: SQLite-backed persistent memory storing learned subject-predicate-object knowledge facts and past conversational messages.
4. **Tool Registry & Security Guard**: Enforces tiered permissions (SAFE, CAUTION, DANGEROUS) before any local system action or file operation is executed.
