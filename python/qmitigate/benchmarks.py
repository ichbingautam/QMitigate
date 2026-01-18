"""
Benchmarking utilities for QMitigate

Demonstrates performance advantages of C++ implementation over pure Python.
"""

import time
from typing import Optional, Tuple
import numpy as np


def run_benchmark(
    num_qubits: int = 15,
    circuit_depth: int = 50,
    num_runs: int = 5
) -> dict:
    """
    Benchmark QMitigate C++ engine performance.

    Args:
        num_qubits: Number of qubits (default 15 for reasonable runtime)
        circuit_depth: Number of random gates
        num_runs: Number of benchmark runs for averaging

    Returns:
        Dictionary with timing and statistics
    """
    try:
        from .qmitigate_cpp import Circuit, Simulator
    except ImportError:
        return {"error": "C++ module not built"}

    # Build a random circuit
    circuit = Circuit(num_qubits)
    rng = np.random.default_rng(42)

    for _ in range(circuit_depth):
        gate_type = rng.integers(0, 4)
        qubit = rng.integers(0, num_qubits)

        if gate_type == 0:
            circuit.h(qubit)
        elif gate_type == 1:
            circuit.x(qubit)
        elif gate_type == 2:
            circuit.rx(qubit, rng.random() * np.pi)
        elif gate_type == 3 and num_qubits > 1:
            target = (qubit + 1) % num_qubits
            circuit.cnot(qubit, target)

    sim = Simulator()

    # Warm-up run
    _ = sim.run(circuit)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        state = sim.run(circuit)
        end = time.perf_counter()
        times.append(end - start)

    state_size_mb = (2 ** num_qubits * 16) / (1024 * 1024)

    return {
        "num_qubits": num_qubits,
        "circuit_depth": circuit_depth,
        "state_vector_size_mb": state_size_mb,
        "mean_time_seconds": float(np.mean(times)),
        "std_time_seconds": float(np.std(times)),
        "min_time_seconds": float(np.min(times)),
        "max_time_seconds": float(np.max(times)),
        "gates_per_second": circuit_depth / np.mean(times)
    }


def compare_with_qiskit(
    num_qubits: int = 10,
    circuit_depth: int = 20
) -> dict:
    """
    Compare QMitigate performance with Qiskit Aer.

    Args:
        num_qubits: Number of qubits
        circuit_depth: Circuit depth

    Returns:
        Comparison metrics
    """
    results = {"num_qubits": num_qubits, "circuit_depth": circuit_depth}

    # Benchmark QMitigate
    try:
        from .qmitigate_cpp import Circuit, Simulator

        circuit = Circuit(num_qubits)
        rng = np.random.default_rng(42)
        for _ in range(circuit_depth):
            q = rng.integers(0, num_qubits)
            circuit.h(q)
            if num_qubits > 1:
                circuit.cnot(q, (q + 1) % num_qubits)

        sim = Simulator()
        _ = sim.run(circuit)  # Warm up

        start = time.perf_counter()
        state = sim.run(circuit)
        qmitigate_time = time.perf_counter() - start
        qmitigate_probs = state.get_probabilities()

        results["qmitigate"] = {
            "time_seconds": qmitigate_time,
            "available": True
        }
    except ImportError:
        results["qmitigate"] = {"available": False}
        qmitigate_probs = None

    # Benchmark Qiskit if available
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer

        qc = QuantumCircuit(num_qubits)
        rng = np.random.default_rng(42)
        for _ in range(circuit_depth):
            q = rng.integers(0, num_qubits)
            qc.h(q)
            if num_qubits > 1:
                qc.cx(q, (q + 1) % num_qubits)

        sim = Aer.get_backend('statevector_simulator')

        start = time.perf_counter()
        job = sim.run(qc)
        result = job.result()
        qiskit_time = time.perf_counter() - start
        qiskit_sv = np.asarray(result.get_statevector())
        qiskit_probs = np.abs(qiskit_sv) ** 2

        results["qiskit"] = {
            "time_seconds": qiskit_time,
            "available": True
        }

        # Check correctness
        if qmitigate_probs is not None:
            qmitigate_arr = np.array(qmitigate_probs)
            fidelity = np.sum(np.sqrt(qmitigate_arr * qiskit_probs)) ** 2
            results["fidelity"] = float(fidelity)
            results["speedup"] = qiskit_time / qmitigate_time

    except ImportError:
        results["qiskit"] = {"available": False}

    return results


def scaling_benchmark(max_qubits: int = 20) -> list:
    """
    Measure scaling behavior across qubit counts.

    Returns list of benchmark results for 5, 10, 15, ... qubits.
    """
    results = []
    for n in range(5, max_qubits + 1, 5):
        try:
            result = run_benchmark(num_qubits=n, circuit_depth=20, num_runs=3)
            results.append(result)
            print(f"Qubits: {n:2d} | Time: {result['mean_time_seconds']:.4f}s | "
                  f"State: {result['state_vector_size_mb']:.1f} MB")
        except Exception as e:
            print(f"Qubits: {n:2d} | Error: {e}")
            break
    return results
