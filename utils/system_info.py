"""Collect OS / Python / GPU / RAM info for run manifests.

Used by `utils.run_logger` to record the environment a batch was run in,
so future debugging across machines is traceable.
"""

import platform
import psutil


def gather_system_info() -> dict:
    """Return a dict with os, python_version, gpu, system_ram_gb."""
    return {
        'os': f"{platform.system()} {platform.release()}",
        'python_version': platform.python_version(),
        'gpu': _detect_gpu(),
        'system_ram_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }


def _detect_gpu() -> dict:
    """Return {'name': str|None, 'vram_gb': float}. Falls back gracefully."""
    try:
        import torch  # imported lazily — torch is a heavy import
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                'name': torch.cuda.get_device_name(0),
                'vram_gb': round(props.total_memory / (1024 ** 3), 1),
            }
    except Exception:
        pass
    return {'name': None, 'vram_gb': 0.0}
