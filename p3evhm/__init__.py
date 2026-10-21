"""P3-EVHM: Parallel Physics-Informed Prognostics for EV Battery Health Management.

A small, dependency-light framework for lithium-ion battery SoH estimation and
RUL prediction that combines:
  - physics-informed LSTM training (data loss + monotonicity + temperature prior),
  - particle-filter RUL estimation with prediction intervals,
  - a 6-baseline benchmark harness (B1-B6), and
  - GPU acceleration through PyTorch CUDA operations (device="cuda").

Quickstart (no data needed):
    python -m p3evhm.demo --source synthetic
Quickstart with NASA PCoE .mat files in ./data:
    python -m p3evhm.demo --source nasa --cell B0005
"""

__version__ = "1.0.0"
__all__ = ["data", "models", "evaluate"]
