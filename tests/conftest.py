"""Add the project root to sys.path so tests can import extractors/utils."""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# MKL on AMD CPUs crashes when using the default Intel-thread layer for
# floating-point BLAS calls (Windows, Ryzen). Force sequential threading so
# matrix-multiply in librosa/numpy works without a fatal exception (0xc06d007f).
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")
