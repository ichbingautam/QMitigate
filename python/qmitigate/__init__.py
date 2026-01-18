"""
QMitigate: High-Performance Quantum Error Mitigation Engine

A C++ quantum state-vector simulator with Python bindings,
featuring Zero-Noise Extrapolation (ZNE) for NISQ error mitigation.

Author: Shubham Gautam
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Shubham Gautam"

try:
    from .qmitigate_cpp import (
        QuantumState,
        Circuit,
        Simulator,
        NoiseModel,
        GateType,
        NoiseChannel,
        gates,
        create_bell_state,
        create_ghz_state,
        create_w_state,
        create_depolarizing_noise_model,
        fold_circuit_global,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import qmitigate_cpp. Build the C++ extension first:\n"
        "  mkdir build && cd build && cmake .. && make\n"
        f"Original error: {e}"
    )

from .zne import (
    ZeroNoiseExtrapolator,
    richardson_extrapolate,
    linear_extrapolate,
    exponential_extrapolate,
)

from .benchmarks import run_benchmark, compare_with_qiskit

__all__ = [
    # Core C++ classes
    "QuantumState",
    "Circuit",
    "Simulator",
    "NoiseModel",
    "GateType",
    "NoiseChannel",
    "gates",
    # Factory functions
    "create_bell_state",
    "create_ghz_state",
    "create_w_state",
    "create_depolarizing_noise_model",
    "fold_circuit_global",
    # ZNE
    "ZeroNoiseExtrapolator",
    "richardson_extrapolate",
    "linear_extrapolate",
    "exponential_extrapolate",
    # Benchmarks
    "run_benchmark",
    "compare_with_qiskit",
]
