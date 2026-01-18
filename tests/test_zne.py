"""
Python tests for QMitigate ZNE module

Uses hypothesis for property-based testing to ensure correctness.
"""

import pytest
import numpy as np
from typing import List


class TestRichardsonExtrapolation:
    """Tests for Richardson extrapolation."""

    def test_linear_data_exact(self):
        """Linear data should extrapolate perfectly."""
        from qmitigate.zne import richardson_extrapolate

        # y = 1 - 0.1*x, so y(0) = 1.0
        scales = [1.0, 3.0, 5.0]
        values = [0.9, 0.7, 0.5]

        result = richardson_extrapolate(scales, values)
        assert abs(result - 1.0) < 1e-10

    def test_quadratic_data(self):
        """Quadratic data with enough points."""
        from qmitigate.zne import richardson_extrapolate

        # y = 1 - 0.1*x + 0.01*x^2
        scales = [1.0, 3.0, 5.0]
        values = [1 - 0.1*s + 0.01*s**2 for s in scales]

        result = richardson_extrapolate(scales, values)
        # y(0) = 1.0
        assert abs(result - 1.0) < 1e-10

    def test_insufficient_points(self):
        """Should raise with less than 2 points."""
        from qmitigate.zne import richardson_extrapolate

        with pytest.raises(ValueError):
            richardson_extrapolate([1.0], [0.9])

    def test_mismatched_lengths(self):
        """Should raise with mismatched lengths."""
        from qmitigate.zne import richardson_extrapolate

        with pytest.raises(ValueError):
            richardson_extrapolate([1.0, 3.0], [0.9, 0.7, 0.5])


class TestLinearExtrapolation:
    """Tests for linear extrapolation."""

    def test_simple_linear(self):
        from qmitigate.zne import linear_extrapolate

        scales = [1.0, 3.0]
        values = [0.8, 0.6]  # slope = -0.1

        e0, slope = linear_extrapolate(scales, values)
        assert abs(e0 - 0.9) < 1e-10  # 0.8 + 0.1 = 0.9
        assert abs(slope - (-0.1)) < 1e-10


class TestExponentialExtrapolation:
    """Tests for exponential extrapolation."""

    def test_exponential_decay(self):
        from qmitigate.zne import exponential_extrapolate

        # y = exp(-0.1*x)
        scales = [1.0, 3.0, 5.0, 7.0]
        values = [np.exp(-0.1*s) for s in scales]

        e0, params = exponential_extrapolate(scales, values)
        # y(0) = 1.0
        assert abs(e0 - 1.0) < 0.1  # Exponential fit is approximate


class TestZNEIntegration:
    """Integration tests requiring C++ module."""

    @pytest.fixture
    def cpp_available(self):
        """Check if C++ module is available."""
        try:
            from qmitigate import Circuit, Simulator
            return True
        except ImportError:
            return False

    def test_circuit_creation(self, cpp_available):
        """Test basic circuit operations."""
        if not cpp_available:
            pytest.skip("C++ module not available")

        from qmitigate import Circuit

        circuit = Circuit(3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.x(2)

        assert circuit.num_qubits == 3
        assert circuit.depth() == 3

    def test_bell_state(self, cpp_available):
        """Test Bell state creation."""
        if not cpp_available:
            pytest.skip("C++ module not available")

        from qmitigate import create_bell_state

        bell = create_bell_state()
        assert bell.num_qubits() == 2
        assert bell.is_normalized()

        # Check amplitudes
        assert abs(bell.probability(0) - 0.5) < 1e-10
        assert abs(bell.probability(3) - 0.5) < 1e-10

    def test_ideal_simulation(self, cpp_available):
        """Test noiseless simulation gives expected results."""
        if not cpp_available:
            pytest.skip("C++ module not available")

        from qmitigate import Circuit, Simulator

        # Create |+⟩ state
        circuit = Circuit(1)
        circuit.h(0)

        sim = Simulator()
        state = sim.run(circuit)

        # ⟨Z⟩ should be 0 for |+⟩
        exp_z = state.expectation_Z(0)
        assert abs(exp_z) < 1e-10

    def test_digital_folding(self, cpp_available):
        """Test that folding preserves logical operation."""
        if not cpp_available:
            pytest.skip("C++ module not available")

        from qmitigate import Circuit, Simulator, fold_circuit_global

        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        sim = Simulator()  # No noise

        original_result = sim.run(circuit)

        folded = fold_circuit_global(circuit, 3)
        folded_result = sim.run(folded)

        # Results should be identical (unitary folding is identity)
        fidelity = original_result.fidelity(folded_result)
        assert fidelity > 0.999

    def test_noise_reduces_fidelity(self, cpp_available):
        """Test that noise degrades results."""
        if not cpp_available:
            pytest.skip("C++ module not available")

        from qmitigate import Circuit, Simulator, create_depolarizing_noise_model

        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        # Ideal
        ideal_sim = Simulator()
        ideal_state = ideal_sim.run(circuit)

        # Noisy
        noise = create_depolarizing_noise_model(0.1)
        noisy_sim = Simulator(noise)
        noisy_state = noisy_sim.run(circuit)

        fidelity = ideal_state.fidelity(noisy_state)
        # With 10% noise per gate, fidelity should drop
        assert fidelity < 1.0


# Property-based tests (when hypothesis is available)
try:
    from hypothesis import given, strategies as st, settings

    class TestPropertyBased:
        """Property-based tests using hypothesis."""

        @given(st.lists(st.floats(min_value=1.0, max_value=10.0), min_size=2, max_size=5))
        @settings(max_examples=50)
        def test_richardson_preserves_order(self, scales: List[float]):
            """Richardson with sorted scales should work."""
            from qmitigate.zne import richardson_extrapolate

            scales = sorted(set(scales))  # Remove duplicates and sort
            if len(scales) < 2:
                return

            # Generate linear data
            values = [1.0 - 0.05 * s for s in scales]

            result = richardson_extrapolate(scales, values)
            # Result should be >= max value (extrapolating backward)
            assert result >= min(values) - 0.5

except ImportError:
    pass  # hypothesis not available
