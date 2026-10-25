# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""ONNX Runtime utilities for UniFace.

This module provides helper functions for creating and managing ONNX Runtime
inference sessions with automatic hardware acceleration detection.
"""

from __future__ import annotations

import functools

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        'onnxruntime is not installed. Install it with one of:\n'
        '  pip install uniface[cpu]   # CPU / Apple Silicon\n'
        '  pip install uniface[gpu]   # NVIDIA GPU (CUDA)\n'
        'Do not install both — they conflict.'
    ) from e

import ctypes
import os

from uniface.log import Logger


# Load order matches the CUDA provider's dependencies. cuBLAS must see its
# Lt library first. These wheels ship the libraries inside site-packages, which
# is not on the system linker path of a machine that only has the NVIDIA driver.
_CUDA_LIBRARY_SONAMES = (
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libnvrtc.so.12",
    "libcudnn.so.9",
)


def _preload_cuda_libraries() -> None:
    """Load pip CUDA libraries so ONNX Runtime can open the CUDA provider.

    A CPU-only install has no nvidia package, so this stays quiet. RTLD_GLOBAL
    keeps each soname visible when the provider is opened later.
    """
    try:
        import nvidia
    except Exception:
        return

    wanted = set(_CUDA_LIBRARY_SONAMES)
    found: dict[str, str] = {}
    for root in nvidia.__path__:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename in wanted and filename not in found:
                    found[filename] = os.path.join(dirpath, filename)

    for soname in _CUDA_LIBRARY_SONAMES:
        library_path = found.get(soname)
        if library_path is None:
            continue
        try:
            ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            Logger.warning("CUDA library could not be loaded: %s", library_path)


_preload_cuda_libraries()

__all__ = ['create_onnx_session', 'get_available_providers']


@functools.lru_cache(maxsize=1)
def get_available_providers() -> list[str]:
    """Get list of available ONNX Runtime execution providers.

    Automatically detects and prioritizes hardware acceleration:
    - CoreML on Apple Silicon (M1/M2/M3/M4)
    - CUDA on NVIDIA GPUs
    - CPU as fallback (always available)

    Returns:
        Ordered list of execution providers to use.

    Example:
        >>> providers = get_available_providers()
        >>> # On macOS: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        >>> # On Linux with CUDA: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    """
    available = ort.get_available_providers()
    providers = []

    # Priority order: CoreML > CUDA > CPU
    if 'CoreMLExecutionProvider' in available:
        providers.append('CoreMLExecutionProvider')
        Logger.info('CoreML acceleration enabled (Apple Silicon)')

    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
        Logger.info('CUDA acceleration enabled (NVIDIA GPU)')

    # CPU is always available as fallback
    providers.append('CPUExecutionProvider')

    if len(providers) == 1:
        Logger.info('Using CPU execution (no hardware acceleration detected)')

    return providers


def create_onnx_session(
    model_path: str,
    providers: list[str] | None = None,
) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session with optimal provider selection.

    Args:
        model_path: Path to the ONNX model file.
        providers: List of execution providers to use. If None, automatically
            detects best available providers.

    Returns:
        Configured ONNX Runtime session.

    Raises:
        RuntimeError: If session creation fails.

    Example:
        >>> session = create_onnx_session('model.onnx')
        >>> # Automatically uses best available providers

        >>> session = create_onnx_session('model.onnx', providers=['CPUExecutionProvider'])
        >>> # Force CPU-only execution
    """
    if providers is None:
        providers = get_available_providers()

    # Suppress ONNX Runtime warnings (e.g., CoreML partition warnings)
    # Log levels: 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Only show ERROR and FATAL

    try:
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        active_provider = session.get_providers()[0]
        Logger.debug(f'Session created with provider: {active_provider}')

        # DEBUG level: only visible when the user opts in via enable_logging(logging.DEBUG)
        provider_names = {
            'CoreMLExecutionProvider': 'CoreML (Apple Silicon)',
            'CUDAExecutionProvider': 'CUDA (NVIDIA GPU)',
            'CPUExecutionProvider': 'CPU',
        }
        provider_display = provider_names.get(active_provider, active_provider)
        Logger.debug(f'Model loaded from {model_path} ({provider_display})')

        return session
    except Exception as e:
        Logger.error(f'Failed to create ONNX session: {e}', exc_info=True)
        raise RuntimeError(f'Failed to initialize ONNX Runtime session: {e}') from e
